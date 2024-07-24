
#include "include_all.h"
#include "utils.h"
#include "network.h"
#include "rdma_shim.cuh"
#define BUFFER_SIZE 100'000'000


// #include "rdma.h"
// #include "rdma_shim.cuh"
// extern "C" {
// #include "rdma_shim.h"
// }
// #include <cstdlib>
// #include <cstring>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <signal.h>
//#include <infiniband/mlx5dv.h>
//#include <device/pt-to-pt/utils_device.h>

namespace po = boost::program_options;
std::string mode;
std::string mode_cqe;
std::string remote_address;
std::string data_location;
std::string buffer_location;
std::string local_address;
int batch;
uint64_t n;
int write_size;
bool quiet;
int gpu_batch;
uint64_t max_rate;
uint64_t send_time = 0;
uint64_t cqe_time = 0;
uint64_t cqe_count = 0;
uint64_t sleep_us = 0;
uint64_t errors = 0;
uint64_t max_errors = 1000;
bool stop = false;
uint64_t start = 0;

int arg_parse(int argc, char **argv, po::variables_map &vm) {
    // clang-format off
    po::options_description desc("RDMA from GPU traffic generator");
            desc.add_options()
            ("help", "Print help message")
            ("mode", po::value<std::string>(&mode)->default_value("gpu"), "Mode of operation: standard, crafted or gpu")
            ("consume-cqe", po::value<std::string>(&mode_cqe)->default_value("gpu"), "How to consume cqe: standard, crafted or gpu")
            ("driver-location", po::value<std::string>(&data_location)->default_value("cudaMallocManaged"), "How to allocate rdma-core data: cudaMallocHost, host, cudaMallocManaged")
            ("buffer-location", po::value<std::string>(&buffer_location)->default_value("cuda"), "Where to allocate the buffer for RDMA: cuda or host")
            ("batch", po::value<int>(&batch)->default_value(2048), "How often to post signaled and consume cqe")
            ("number", po::value<uint64_t>(&n)->default_value(10000), "How many requests to send, 0 for infinite")
            ("write-size", po::value<int>(&write_size)->default_value(1000), "Buffer size to write")
            ("quiet", po::value<bool>(&quiet)->default_value(0), "Be less verbose")
            ("max-rate", po::value<uint64_t>(&max_rate)->default_value(100), "Maximum rate to send at (or try to), in Gbps. Does not consider headers!")
            ("sleep-us", po::value<uint64_t>(&sleep_us)->default_value(0), "us to sleep between each inference. Use to avoid 'error 12' when running too-fast on CPU.")
            ("remote-address", po::value<std::string>(&remote_address)->default_value("192.168.128.1"), "Remote address for the RDMA packets")
            ("local-address", po::value<std::string>(&local_address)->default_value(""), "Local address for the RDMA packets. Will be used also to pick the correct card!")
            ("gpu-batch", po::value<int>(&gpu_batch)->default_value(1), "Number of requests to send ayncrhonously when running from GPU.");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    // clang-format on

    if (vm.count("help")) {
        desc.print(std::cout);
        return 1;
    }
    std::vector<std::string> modes{"standard", "crafted", "gpu"};
    if (std::find(modes.begin(), modes.end(), mode) == modes.end()) {
        printf("Error: mode %s is not supported\n", mode.c_str());
        return 1;
    }
    if (std::find(modes.begin(), modes.end(), mode_cqe) == modes.end()) {
        printf("Error: cqe consumption mode %s is not supported\n",
               mode_cqe.c_str());
        return 1;
    }

    return 0;
}



__device__ uint64_t gpu_start_time = 0;
__device__ uint64_t gpu_exec_time_sum = 0;
__device__ uint64_t gpu_cqe_time_sum = 0;
__device__ uint64_t gpu_exec_count = 0;
__device__ uint64_t gpu_cqe_count = 0;

__global__ void set_start_time() { gpu_start_time = clock(); }
__global__ void sum_time_cqe() {
    gpu_cqe_time_sum += (clock() - gpu_start_time); 
    gpu_cqe_count++;
}
__global__ void sum_time() {
    gpu_exec_time_sum += (clock() - gpu_start_time); 
    gpu_exec_count++;
}
__global__ void sum_time_multi(int batch) {
    gpu_exec_time_sum += (clock() - gpu_start_time); 
    gpu_exec_count+=batch;
}

void on_exit(int signal)
{
    stop = true;
}


uint64_t last_check = 0;
uint64_t next_check = 0;
uint64_t last_count = 0;
uint64_t start_to_send = 0;
uint64_t next_print = 0;

inline void rate_limiter(uint64_t sent_pkts, bool print, uint64_t interval = 10'000, uint64_t print_interval = 1'000'000'000)
{
    uint64_t t = now();
    if(start_to_send == 0)
    {
        start_to_send = t;
        next_print = t+print_interval;
        next_check = t+1;
    }

    if (t < next_check)
        return;

    // This does not consider headers!
    double rate = write_size * 8.0 *(sent_pkts - last_count) / (1.0*t - last_check);

    int sleeps = 0;
    int usleep_time = 1;
    while( rate > max_rate)
    {
        sleeps++;
        usleep(usleep_time);
        // if (sleeps%1000) // We want to print every 100 times we slept
        //     printf("Rate exceed! (slept %i ns so far)\n", usleep_time*sleeps);
        rate = write_size * 8 *(sent_pkts - last_count) / (1.0*now() - last_check);
    }

    last_check = now();
    next_check = last_check + interval;
    last_count = sent_pkts;

    if(t > next_print)
    {
        double avg_rate = write_size * 8.0 * sent_pkts / (t - start_to_send);
        printf("%li-RESULT-SEND_RATE %f\n", t, rate*1e9);
        printf("%li-RESULT-AVG_SEND_RATE %f\n", t, avg_rate*1e9);
        next_print = t + print_interval;
    }

}


int main(int argc, char **argv) {
    signal(SIGINT, on_exit);
    really_now(&start);
    po::variables_map vm;
    int ret = arg_parse(argc, argv, vm);
    if (ret) return 1;

    cudaStream_t stream;

    bool use_gpu_rdma = (mode == "gpu");
    bool use_gpu = (mode == "gpu" || mode_cqe == "gpu" || data_location == "cuda" || buffer_location != "host");

    if (use_gpu) {
        cudaSetDevice(0);
        cudaStreamCreate(&stream);
    }

    // We pre-alloc an area of memory, so that we have to register only that!
    // And we may even alloc it in CUDA
    size_t driver_data_size = 1 * 1024 * 1024 * 1204;
    void *driver_data = nullptr;

    // Test flags
    if (data_location == "cudaMallocHostZ")
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,0));
    if (data_location == "cudaMallocHostP")
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                    cudaHostAllocPortable));
    if (data_location == "cudaMallocHostM")
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                    cudaHostAllocMapped));
    if (data_location == "cudaMallocHostPM")
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                    cudaHostAllocPortable|cudaHostAllocMapped));
    if (data_location == "cudaMallocHostW")
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                    cudaHostAllocWriteCombined));
    if (data_location == "cudaMallocHostPW")
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                    cudaHostAllocPortable|cudaHostAllocWriteCombined));
    if (data_location == "cudaMallocHostMW")
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                    cudaHostAllocMapped|cudaHostAllocWriteCombined));
    if (data_location == "cudaMallocHostPMW")
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                    cudaHostAllocMapped|cudaHostAllocPortable|cudaHostAllocWriteCombined));

    if (driver_data == nullptr){
        if (data_location == "cudaMallocHost")
        {
            LOG(INFO) << "driver data is being allocated through cudaMallocHost";
            CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                        cudaHostAllocPortable | cudaHostAllocMapped |
                        cudaHostAllocWriteCombined));
        }
        if (data_location == "cudaMallocManaged")
        {
            LOG(INFO) << "driver data is being allocated through cudaManaged";
            CUDA_CALL(cudaMallocManaged(&driver_data, driver_data_size));
        }
        else
        {
            if (data_location != "host")
                LOG(WARNING) << "Unsupported data_location " << data_location << ": going for the default";
            LOG(INFO) << "driver data is being allocated through malloc";
            driver_data = malloc(driver_data_size);

        }
    }

    // Tell rdma-core to use our custom allocators
    setup_custom_allocs(driver_data, driver_data_size);

    // Init rdma
    std::vector<std::string> some_ips(
        {"192.168.128.1", "192.168.129.1", "192.168.130.1", "192.168.134.1", "192.168.200.34"});
    if (local_address != "")
    {
        some_ips.clear();
        some_ips.push_back(local_address);
    }
    ibv_qp * qp;
    ibv_pd * pd = nullptr;
    CHECK(init_rdma(&qp, &pd, some_ips, remote_address) == 0) << "error while initializing RDMA stack!";

    // Memory stuff
    uint8_t *gpu_buffer;
    if (buffer_location == "host")
    {
        unsigned int flag = 1;
        gpu_buffer = (uint8_t *)malloc(BUFFER_SIZE);
        ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                    (CUdeviceptr)gpu_buffer);
    }
    else
        {
        CUDA_CALL(cudaMalloc(&gpu_buffer, BUFFER_SIZE));
    }
    CHECK(gpu_buffer != nullptr) << "error while allocating memory";

    auto mr = ibv_reg_mr(qp->pd, gpu_buffer, BUFFER_SIZE,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    CHECK(mr) << "Error while registering memory";

    struct rdma_shim_data *data = (struct rdma_shim_data *)rdma_shim_malloc(
        sizeof(struct rdma_shim_data));

    // --- Register memory
    void *l, *h;
    uint64_t total = prepare_rdma(qp, data, &l, &h);

    if (use_gpu) {
        if (data_location == "host")
            register_cuda_driver_data(driver_data, driver_data_size);
        register_cuda_areas(data);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    // // do_standard_write(qp, 0xaa0011, nullptr,0, mr->lkey);

    ibv_wc wc;

    uint64_t total_pkts = 0;
    uint64_t start;
    uint64_t t;

    if (gpu_batch == 1)
    {
        for (uint64_t i = 0; (i < n || n == 0) && !stop; i++) {
            if (quiet)
            {
                if ((i % 10000) == 0)
                    LOG(INFO) << i;
            }
            else
                LOG(INFO) << i;

            rate_limiter(total_pkts, true);
            really_now(&t);


            if ((i % batch) == 0 && i != 0) {
                start = 0;
                if (mode_cqe == "standard")
                    poll_single_cq_standard(qp->send_cq, &wc);
                else if (mode_cqe == "gpu")
                {

                    set_start_time<<<1,1>>>();
                    consume_cqe_kernel<<<1, 1, 1>>>(data);
                    sum_time_cqe<<<1,1>>>();
                }
                else
                    consume_send_cq(data);
                really_now(&t);
                cqe_time+=(t-start);
                cqe_count++;
            }
            really_now(&start);
            if (mode == "standard")
                ret = do_standard_write(qp, 0xff | (i << 16), gpu_buffer,
                                        write_size, mr->lkey, (i % batch) == 0, quiet);
            else if (mode == "gpu")
            {
                set_start_time<<<1,1>>>();
                ret = do_gpu_crafted_write(data, 0xff | (i << 16), gpu_buffer,
                                           write_size, mr->lkey, (i % batch) == 0, quiet);
                sum_time<<<1,1>>>();
            }
            else
                ret = do_crafted_write(data, 0xff | (i << 16), gpu_buffer,
                                       write_size, mr->lkey, (i % batch) == 0, quiet);
            really_now(&t);
            send_time+=(t - start);
            total_pkts++;
            if (sleep_us >0)
                usleep(sleep_us);
            if (ret != 0)
            {
                LOG(WARNING) << "Error: verb returned " << ret;
                errors++;
                if (errors > max_errors)
                {
                    LOG(ERROR) << "Too many errors!";
                    break;
                }
            }
        }
    }
    else 
    {
        if (mode != "gpu" || mode_cqe != "gpu")
            LOG(FATAL) << "This combination of parameters is not valid!";

        for(uint64_t i = 0; i< n && !stop; i+= gpu_batch)
        {
            if (!quiet) // || i%1024 == 0)
                LOG(INFO) << "Posting " << gpu_batch << " requests";
            set_start_time<<<1,1>>>();
            rdma_write_with_imm_kernel_multiple<<<1, 1, 1>>>(data,
                                            (void *)mr->addr,     // buffer
                                            write_size,               // size
                                            mr->lkey,               // lk
                                            0xbbaabbaa,         // rk
                                            (void *)0xffff1234, // raddr
                                            i,
                                            gpu_batch,
                                            batch);
            sum_time_multi<<<1,1>>>(gpu_batch);
            total_pkts+=gpu_batch;
            rate_limiter(total_pkts, true);
        }
    }

    if (use_gpu) CUDA_CALL(cudaDeviceSynchronize());
    if (mode == "gpu")
    {
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
        printf("RESULT-CUDA_CLOCK %f\n", prop.clockRate/1e3);

        uint64_t send_time_sum;
        uint64_t send_count;

        CUDA_CALL(cudaMemcpyFromSymbol(&send_time_sum, gpu_exec_time_sum, sizeof(send_time_sum), 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&send_count, gpu_exec_count, sizeof(send_count), 0, cudaMemcpyDeviceToHost));
        // LOG(INFO) << "SUM " << send_time_sum;
        // LOG(INFO) << "COUNT " << send_count;
        // LOG(INFO) << "SO  " <<send_time_sum << "/ (1e-6 * " << send_count<< " * " << prop.clockRate;

        double send_time_gpu_avg = send_time_sum / (1e-6 * send_count * prop.clockRate);
        printf("RESULT-AVG_SEND %f\n", send_time_gpu_avg);
    }
    else
    {
        printf("RESULT-AVG_SEND %f\n", send_time /(double) n);
    }

    if (mode_cqe == "gpu")
    {
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
        printf("RESULT-CUDA_CLOCK %f\n", prop.clockRate/1e3);

        uint64_t cqe_time_sum;
        uint64_t cqe_count;

        CUDA_CALL(cudaMemcpyFromSymbol(&cqe_time_sum, gpu_cqe_time_sum, sizeof(cqe_time_sum), 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&cqe_count, gpu_cqe_count, sizeof(cqe_count), 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        double cqe_time_gpu_avg = cqe_time_sum / (1e-6 * cqe_count * prop.clockRate);
        printf("RESULT-AVG_CQE %f\n", cqe_time_gpu_avg);
    }
    else
    {
        printf("RESULT-AVG_CQE %f\n", cqe_time /(double) cqe_time);
    }

    printf("RESULT-RUNTIME %li\n", now() - start);
    LOG(INFO) << "Hejdȧ";

    return 0;
}
