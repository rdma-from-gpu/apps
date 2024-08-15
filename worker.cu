#include "include_all.h"
#include "utils.h"
#include "network.h"
#include "buffers.h"
#include "executor.h"
#include "rdma_shim.cuh"
#include "grpc.h"


#define STRING(x) #x
#define XSTRING(x) STRING(x)


namespace po = boost::program_options;


// Paramers we will parse from CMDLINE:
std::string mode;
// std::string remote_address;
std::string data_location;
std::string buffer_location;
bool quiet;
// int gpu_batch;
int batch;
uint64_t gpu_buffer_size;
int port;
int n_slots;
int slot_size;
int input_size;
int output_size;
bool stop = false;
uint64_t warmup_rounds;
std::string workload;
bool preload = false;
std::string modelzoo;
bool async_metrics;
uint64_t metrics_size;
uint64_t metrics_interval;
executor_data *executor;
bool random_slots;
int sleep_start;
int profile;
int profile_concurrency;
int max_runtime;
uint64_t start;
bool poll;
bool copy_io;
uint64_t max_inferences;
int copy_mode;
bool reported_metrics = false;

std::vector<void *> gpu_slots;
std::vector<void *> cpu_slots;

std::thread *killer_thread_ptr;

std::string get_default_modelzoo() {
    std::filesystem::path source_dir =
        XSTRING(SOURCE_ROOT); // We define this in the CMake config
    auto base = std::filesystem::absolute(source_dir);
    // First time look in the compile folder, then in current folder. Or maybe
    // should we do the opposite?
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 5; i++) // Look up to 5 parents
        {
            LOG(INFO) << "Searching for modelzoo in " << base;
            if (std::filesystem::exists(base / "modelzoo"))
                return std::string(base / "modelzoo");
            else
                base = base.parent_path();
        }
        base = std::filesystem::absolute(std::filesystem::path("."));

    }

    LOG(FATAL) << "No modelzoo folder found!";
    return ".";
}

int arg_parse(int argc, char **argv, po::variables_map &vm) {
    // clang-format off
    po::options_description desc("RDMA from GPU: worker");
            desc.add_options()
            ("help", "Print help message")
            ("mode", po::value<std::string>(&mode)->default_value("gpu-gpu"), "Mode of operation: cpu-cpu,gpu-cpu,cpu-gpu,gpu-gpu")
            ("driver-location", po::value<std::string>(&data_location)->default_value("cudaMallocHost"), "How to allocate rdma-core data: cudaMallocHost, malloc, cudaMallocManaged")
            ("buffer-location", po::value<std::string>(&buffer_location)->default_value("cudaMalloc"), "Where to allocate the buffer for RDMA: cudaMalloc, malloc, cudaMallocManaged")
            ("gpu-buffer-size", po::value<uint64_t>(&gpu_buffer_size)->default_value(150*1024*1024), "Size of the GPU registered memory") // Remember on T4 we must be below 230MB
            ("slots", po::value<int>(&n_slots)->default_value(2000), "How many slots to allocate") // This is just a random prime number
            ("input-size", po::value<int>(&input_size)->default_value(3*224*224*4), "Size of each input") // This is just a random prime number
            ("output-size", po::value<int>(&output_size)->default_value(1000*4), "Size of each output") // This is just a random prime number
            ("port", po::value<int>(&port)->default_value(3333), "Server port")
            ("batch", po::value<int>(&batch)->default_value(2048), "How often to post signaled and consume cqe")
            ("workload", po::value<std::string>(&workload)->default_value("forward"), "What to do?")
            ("warmup", po::value<uint64_t>(&warmup_rounds)->default_value(10000), "How many rounds of warmup to do?")
            ("modelzoo", po::value<std::string>(&modelzoo)->default_value(""), "Modelzoo path, empty to search in parents")
            ("preload", po::value<bool>(&preload)->default_value(false), "Wether to preload or not the (eventual) model")
            ("async-metrics", po::value<bool>(&async_metrics)->default_value(true), "Don't print anything at run-time, collect statistics to print at the end")
            ("metrics-size", po::value<uint64_t>(&metrics_size)->default_value(100'000'000), "If async metrics, how many to store?")
            ("metrics-interval", po::value<uint64_t>(&metrics_interval)->default_value(2000), "How often to print/generate metrics")
            ("random-slots", po::value<bool>(&random_slots)->default_value(false), "Give slots to client(s) in a random order")
            ("start-sleep", po::value<int>(&sleep_start)->default_value(5), "How much to sleep before starting the execution. This should allow the client to warm-up and start writing")
            ("profile", po::value<int>(&profile)->default_value(false), "Don't wait for inputs. Nor send outputs. Just run (1 would run the simple profiler, 2 the concurrent profiler, 3 would save **ALL** execution times in ./times.txt, 4 would launch the graph from CPU - for NSIGHT).")
            ("profile-concurrency", po::value<int>(&profile_concurrency)->default_value(1), "NUmber of concurrent streams to launch")
            ("poll", po::value<bool>(&poll)->default_value(true), "How to handle end of execution? 1 would use the traditional cudaDeviceSynchronize, 0 would use a more relaxed polling")
            //("write-size", po::value<int>(&write_size)->default_value(1000), "Buffer size to write")
            ("quiet", po::value<bool>(&quiet)->default_value(0), "Be less verbose")
            ("max-runtime", po::value<int>(&max_runtime)->default_value(300), "How long to run. Use it to avoid NPF errors!")
            ("copy-mode", po::value<int>(&copy_mode)->default_value(1), "Copy mode for I/O of the model. 0 disables it, 1 memcpy, 2 kernel with for loop.")
            ("max-inferences", po::value<uint64_t>(&max_inferences)->default_value(0), "Max inferences to run. 0 for infinite loop");
            //("remote-address", po::value<std::string>(&remote_address)->default_value("192.168.128.1"), "Remote address for the RDMA packets")
            //("gpu-batch", po::value<int>(&gpu_batch)->default_value(1), "Number of requests to send ayncrhonously when running from GPU.");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    // clang-format on

    if (vm.count("help")) {
        desc.print(std::cout);
        return 1;
    }
    std::vector<std::string> modes{"cpu-cpu", "gpu-cpu", "cpu-gpu", "gpu-gpu" };

    if (std::find(modes.begin(), modes.end(), mode) == modes.end()) {
        printf("Error: mode %s is not supported\n", mode.c_str());
        return 1;
    }



    std::vector<std::string> locations {"cudaMallocHost", "cudaMallocManaged", "cudaMalloc", "malloc" };


    if (std::find(locations.begin(),locations.end(), data_location) ==locations.end()) {
        printf("Error: driver data location %s is not supported\n", data_location.c_str());
        return 1;
    }

    if (std::find(locations.begin(),locations.end(),buffer_location) == locations.end()) {
        printf("Error: buffer data location %s is not supported\n",buffer_location.c_str());
        return 1;
    }

    if (modelzoo == "") modelzoo = get_default_modelzoo();
    LOG(INFO) << "Using modelzoo at " << modelzoo;

    if (copy_mode <0 || copy_mode > 2)
    {
        printf("Error: copy mode %i is not supported\n", copy_mode);
        return 1;
    }


    return 0;
}

using namespace std;



// These prepare the "slots" where the client could write
int create_slots(void *start, uint64_t total_size, int n_slots,
                 int slot_size = 3 * 224 * 224 * 4, bool repeat = false,
                 bool gpu = true) {
    LOG(INFO) << "Creating slots of " << slot_size << " in a " << total_size
              << " area: " << total_size / slot_size << " can fit here";
    LOG(INFO) << "Start from " << start << " for " << (gpu ? "GPU" : "CPU");
    void *current = start;
    void *last = (void *)((((uint64_t)start) + total_size) - slot_size);
    for (int i = 0; i < n_slots; i++) {
        if (current > last)
            if (repeat)
                current = start;
            else
                break;
        current = (void *)(((uint64_t)current) + slot_size);
        if (gpu)
            gpu_slots.push_back(current);
        else
            cpu_slots.push_back(current);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    if (gpu)
        if (random_slots) std::shuffle(gpu_slots.begin(), gpu_slots.end(), g);
    else
        if (random_slots) std::shuffle(cpu_slots.begin(), cpu_slots.end(), g);


    if (gpu)
        LOG(INFO) << "GPU_SLOTS " << gpu_slots[0] << " " << gpu_slots[1] << " " << gpu_slots[2];
    else
        LOG(INFO) << "CPU_SLOTS " << cpu_slots[0] << " " << cpu_slots[1] << " " << cpu_slots[2];

    return gpu_slots.size();
}

std::vector<void *> get_slots(int n, bool gpu = true) {
    // If the client wants less, give it a slice.
    int n_slots = gpu_slots.size();
    if (gpu) {
        if (n < n_slots)
            return std::vector<void *>(gpu_slots.begin(), gpu_slots.begin() + n);
        else
            return gpu_slots;
    } else {
        if (n < n_slots)
            return std::vector<void *>(cpu_slots.begin(), cpu_slots.begin() + n);
        else
            return cpu_slots;
    }

    //    // When we don't have enough, we create a random sequence of element
    //    // Until we fill the n required. It is then responsibility of the
    //    client
    //    // To send them in order
    //    // Hopefully without overcoming n_slots with the "outstanding
    //    requests" std::random_device rd; std::mt19937 g(rd());

    //    std::vector<void*> ret;
    //    for(int i=0; i< n; i+= n_slots)
    //    {
    //        int remaining = (i+n) > n_slots ? n_slots : (n-i);
    //        //ret.emplace_back(slots.begin(), slots.begin() + remaining);
    //        std::copy_n(slots.begin(), ret.at(i), remaining);
    //        std::shuffle(slots.begin(), slots.end(), g);
    //    }
}

void on_exit_signal(int sgnum)
{
    if (!executor->stop || !stop) {
        LOG(INFO)
            << "Politely asking the graph to stop (if it didn't already).";
        stop = true;
        *executor->stop = true;
    }
}
void on_exit() {
    if(!reported_metrics)
        {
            report_metrics(executor);
            reported_metrics = true;
        }
    // if (async_metrics) {
    //     LOG(INFO) << "Waiting for metrics...";
    //     report_metrics(executor);
    // }
    uint64_t stop = now();
    printf("RESULT-RUNTIME %li\n", stop - start);
    CUDA_CALL(cudaDeviceReset());

    // grpc_worker_ptr->shutdown();
    // grpc_thread_ptr->join();
    // killer_thread_ptr->join();
    LOG(INFO) << "Hejdȧ";
}

std::thread self_killer() {
    return std::thread([=]() {
        int i;
        // This is a bad "killable sleep"
        for (i = 0; i < max_runtime && !stop; i++) {
            sleep(1);
        }
        if (i == max_runtime) {
            LOG(INFO) << "Reached max runtime of " << max_runtime << "s.";
            // raise(SIGINT);
            kill(getpid(), SIGINT);
        }
    });
}

__global__ void test_kernel(void * ptr)
{
    // uint64_t* ptr64 = (uint64_t *) ptr;
    // printf("PTR is %li\n", *ptr64);
    // *ptr64=9999876;
}




int main(int argc, char **argv) {
    po::variables_map vm;
    int ret = arg_parse(argc, argv, vm);
    if (ret) return 1;
    start = now();

    cudaStream_t stream;

    cudaSetDevice(0);
    cudaStreamCreate(&stream);

    // We pre-alloc an area of memory, so that we have to register only that!
    // And we may even alloc it in CUDA
    size_t driver_data_size = 1 * 1024 * 1024 * 1204;
    void *driver_data = nullptr;

    // if ( (mode == "gpu-cpu") || (mode == "cpu-gpu"))
    // {
    //     LOG_IF(WARNING, data_location != "cudaMallocHost") << "Ignoring cudaMallocHost for driver data: it must be on CPU!";
    //     data_location = "cudaMallocHost";

    // }

    if (data_location == "cudaMallocHost") {
        LOG(INFO) << "driver data is being allocated through cudaMallocHost";
        CUDA_CALL(cudaMallocHost(&driver_data, driver_data_size,
                                 cudaHostAllocPortable | cudaHostAllocMapped |
                                     cudaHostAllocWriteCombined));
    } else if (data_location == "malloc"){
        LOG(INFO) << "driver data is being allocated through malloc";
        driver_data = malloc(driver_data_size);
    } else if (data_location == "cudaMallocManaged"){
        LOG(INFO) << "driver data is being allocated through cudaMallocManaged";
        CUDA_CALL(cudaMallocManaged(&driver_data, driver_data_size, cudaMemAttachGlobal));

    } else if (data_location == "cuMemCreate")
    {
        LOG(FATAL) << "Until host mapping is supported, we can use this :/ ...";


//             CUmemAllocationProp prop = {};
//             prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
//             prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
//             prop.location.id = 0;
//             // prop.allocFlags=CU_MEM_ALLOCATION_COMP_NONE;
//             size_t granularity;
//             cuMemGetAllocationGranularity(&granularity, &prop,
//                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM);
//             // Ensure size matches granularity requirements for the allocation
//             size_t padded_size = ROUND_UP(driver_data_size, granularity);
//             // Allocate physical memory
//             CUmemGenericAllocationHandle allocHandle;
//             CUresult res = cuMemCreate(&allocHandle, padded_size, &prop, 0);
//             CHECK(res == CUDA_SUCCESS );

//             // Create the address space for it
//             CUdeviceptr ptr;
//             res = (cuMemAddressReserve(&ptr,padded_size, 0, 0, 0)); // alignment = 0 for default alignment
//             CHECK(res == CUDA_SUCCESS );
//             res = cuMemMap(ptr,padded_size , 0, allocHandle, 0);

//             CUmemAccessDesc accessDesc = {};
//             accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
//             accessDesc.location.id = 0;
//             accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

//             // Make the address accessible
//             cuMemSetAccess(ptr,padded_size, &accessDesc, 1);

//             LOG(INFO) << "Allocated driver data at " << ptr;
//             // LOG(INFO) << "Try to access from the CPU";
//             // *((uint64_t*) ptr) =  0x123456;
//             // LOG(INFO) << "Try to access from the CPU ok... Try with a kernel now";

//             // LOG(INFO) << "Pre kernel the value is " << *((uint64_t*)ptr);
//             test_kernel<<<1,1>>>((void*)ptr);
//             LOG(INFO) << "Seems fine...";
//             CUDA_CALL(cudaDeviceSynchronize());
//             LOG(INFO) << "After kernel the value is " << *((uint64_t*)ptr);
//             LOG(INFO) << "Adios";
//             exit(1);
    } else {
            LOG(WARNING) << "Unsupported data_location " << data_location;
            exit(1);
    }

    // Please use our custom allocator!
    setup_custom_allocs(driver_data, driver_data_size);

    // TODO: Here we should check if we want a model workload or something else
    // ALSO TODO: Hard coded path!
    std::string model_path = modelzoo + "/" + workload;
    LOG(INFO) << "Using workload " << workload << " to load a model from "
              << model_path;
    executor = (executor_data *)rdma_shim_malloc(sizeof(executor_data));

    // This should be replaced by a more advanced thing...
    executor->stop = (bool *)rdma_shim_malloc(sizeof(uint64_t));
    
    if (preload) {
        LOG(INFO) << "ASYNC METRICS " << async_metrics << " SO "
                  << (async_metrics ? metrics_size : 0);
        load_model(model_path, executor, async_metrics ? metrics_size : 0,
                   metrics_interval);
        warmup_model(executor, warmup_rounds);
    }

    auto grpc_worker = new GRPCWorker(port);
    auto grpc_thread = grpc_worker->spawn();

    ClientData *client;
    ibv_qp *qp;
    int dst_qp = -1;
    std::string remote_address = "PROFILE";

    if (!profile) {
        while (grpc_worker->clients_.size() == 0) {
            LOG(INFO) << "Waiting for a client to connect...";
            sleep(1);
        }
        LOG(INFO) << "Got a client. Proceed to setup a QP";

        client = grpc_worker->clients_.at(0);
        remote_address = client->client_hello.address();
        dst_qp = client->client_hello.qpnum();
        LOG(INFO) << "Client is at " << remote_address << " with QP num "
                  << dst_qp;

        // Init RDMA stack
        std::vector<std::string> ips({"192.168.128.1", "192.168.129.1",
                                      "192.168.130.1", "192.168.134.1",
                                      "192.168.134.0", "192.168.127.0"});
        ibv_pd *pd = nullptr;
        CHECK(init_rdma(&qp, &pd, ips, remote_address, dst_qp, nullptr,
                        client->client_hello.rc() ? IBV_QPT_RC : IBV_QPT_UC) ==
              0)
            << "error while initializing RDMA stack!";
    }

    // Memory stuff
    uint8_t *gpu_buffer;
    uint8_t *cpu_buffer;
    if (buffer_location == "malloc") {
        LOG(INFO) << "Buffer will be allocated in host space";
        // unsigned int flag = 1;
        cpu_buffer = (uint8_t *)malloc(gpu_buffer_size);
        CUDA_HOST_REGISTER_PRINT((cpu_buffer), gpu_buffer_size,
                             cudaHostRegisterMapped | cudaHostRegisterPortable,
                             "cpu_buffer");

        // ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
        //                             (CUdeviceptr)gpu_buffer);
        // memset(gpu_buffer, 0, gpu_buffer_size);
        gpu_buffer = cpu_buffer;
        LOG(INFO) << "Ok...";
    } else if ( buffer_location == "cudaMallocManaged")
    {
        LOG(INFO) << "Buffer will be allocated in CUDA space, but with self-managed page handling";
        CUDA_CALL(cudaMallocManaged(&gpu_buffer, gpu_buffer_size));
        CUDA_CALL(cudaMemset(gpu_buffer, 0, gpu_buffer_size));
        cpu_buffer = gpu_buffer;
    } else if ( buffer_location == "cudaMallocHost")
    {
        LOG(INFO) << "Buffer will be allocated in CUDA space, but with self-managed page handling";
        CUDA_CALL(cudaMallocHost(&gpu_buffer, gpu_buffer_size));
        CUDA_CALL(cudaMemset(gpu_buffer, 0, gpu_buffer_size));
        cpu_buffer = gpu_buffer;
    } else {
        LOG(INFO) << "Buffer will be allocated in CUDA space";
        CUDA_CALL(cudaMalloc(&gpu_buffer, gpu_buffer_size));
        CUDA_CALL(cudaMemset(gpu_buffer, 0, gpu_buffer_size));
        if (mode != "gpu-gpu")
        {
            cpu_buffer = (uint8_t *)malloc(gpu_buffer_size);
            memset(cpu_buffer, 0, gpu_buffer_size);
            LOG(INFO) << "At least one memory copy is needed when using cpu mediated processing and host memory!";
        }
        else
        {
            LOG(INFO) << "gpu-gpu does not require a CPU buffer!";
            cpu_buffer = nullptr;
        }
    }

    CHECK(gpu_buffer != nullptr) << "error while allocating memory on GPU";
    CHECK(cpu_buffer != nullptr || mode == "gpu-gpu") << "error while allocating memory on CPU";
    if (cpu_buffer != gpu_buffer)
        copy_io = true;
    LOG_IF(INFO, copy_io && (cpu_buffer != nullptr))
        << "The choosen combination of parameters will require at least 1 I/O copy!";

    ibv_mr *mr;
    ibv_mr *cpu_mr = nullptr;
    if (!profile) {
        mr = ibv_reg_mr(qp->pd, gpu_buffer, gpu_buffer_size,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        CHECK(mr) << "Error while registering memory";
        LOG(INFO) << "mr has a length of " << mr->length << "B, so " << mr->addr
                  << "-" << (void *)((uint64_t)mr->addr + mr->length);
        if (mode != "gpu-gpu")
        {
            cpu_mr = ibv_reg_mr(qp->pd, cpu_buffer, gpu_buffer_size,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            CHECK(mr) << "Error while registering memory (CPU)";
            LOG(INFO) << "CPU mr has a length of " << cpu_mr->length << "B, so " << cpu_mr->addr
                      << "-" << (void *)((uint64_t)cpu_mr->addr + cpu_mr->length);
        }

    } else {
        // Just to not change the whole code...
        mr = (ibv_mr *)malloc(sizeof(ibv_mr));
    }

    struct rdma_shim_data *data = (struct rdma_shim_data *)rdma_shim_malloc(
        sizeof(struct rdma_shim_data));

    // --- Register memory
    void *l, *h;
    if (!profile) {
        prepare_rdma(qp, data, &l, &h);

        if (data_location == "malloc")
        {
            LOG(INFO) << "Driver data is '"<< data_location << "': it needs to be registered with CUDA runtime!";
            register_cuda_driver_data(driver_data, driver_data_size);
        }

        register_cuda_areas(data);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    CUDA_CALL(cudaDeviceSynchronize());

    slot_size = request_size(input_size, output_size);
    LOG(INFO) << "Slot size is " << slot_size;
    int aligned_slot_size = aligned_size_32(slot_size);
    LOG_IF(INFO, aligned_slot_size != slot_size)
        << "Slot size was not aligned: " << slot_size << " -> "
        << aligned_slot_size;
    slot_size = aligned_slot_size;

    // Create the slots. This would make more sense when we would have more than
    // 1 client
    if (mode == "gpu-gpu")
    {
        create_slots(gpu_buffer, gpu_buffer_size, n_slots, slot_size, true);
    }
    else
    {
        // In all other cases, we need to create slots for both cpu and gpus
        create_slots(gpu_buffer, gpu_buffer_size, n_slots, slot_size, false, true);
        create_slots(cpu_buffer, gpu_buffer_size, n_slots, slot_size, false, false);
    }


    int client_slots_n;
    std::vector<void *> these_slots;
    if (profile) {
        // We just run over and over on all slots
        client_slots_n = cpu_slots.size();
        these_slots = get_slots(client_slots_n, true);
    } else {
        LOG(INFO) << "Client wants " << client->client_hello.slots()
                  << " slots, we have " << gpu_slots.size();
        client_slots_n = client->client_hello.slots();
        these_slots = get_slots(client_slots_n, (mode == "gpu-gpu") || (mode == "gpu-cpu"));

        // client->worker_hello.mutable_slots()->Reserve(these_slots.size());
        for (int i = 0; i < these_slots.size(); i++) {
            // I'm not beautiful but I work!
            client->worker_hello.mutable_slots()->Add(
                (uint64_t)these_slots.at(i));
        }

        // Unlock the client, sending the data
        client->worker_hello.set_qpnum(qp->qp_num);
        // Here we should get the correct one...
        // client->worker_hello.set_address()

        if ((mode == "gpu-gpu") || (mode == "gpu-cpu"))
        {
            client->worker_hello.set_rkey(mr->rkey);
            client->worker_hello.set_addr((uint64_t)mr->addr);
            client->worker_hello.set_length(mr->length);
        }
        else
        {
            client->worker_hello.set_rkey(cpu_mr->rkey);
            client->worker_hello.set_addr((uint64_t)cpu_mr->addr);
            client->worker_hello.set_length(cpu_mr->length);
        }
        client->worker_hello.set_client_id(
            0); // This is how the worker can re-contact the client when writing
        client->worker_hello.set_slot_size(slot_size);
        client->worker_hello.set_input_size(input_size);
        client->worker_hello.set_output_size(output_size);

        client->worker_data_ready.set_value();

        LOG(INFO) << "Established connection with client " << remote_address
                  << " and QP " << qp->qp_num << " " << dst_qp;
        LOG(INFO) << "Our rkey is " <<  client->worker_hello.rkey() << " and the base address is "
                  << client->worker_hello.addr();
    }
    grpc_worker->shutdown();
    grpc_thread.join();

    // For now, keep a fixed list of slots and copy them at the beginning
    request_t **gpu_requests;
    request_t **cpu_requests;
    CUDA_CALL(cudaMalloc(&gpu_requests, gpu_slots.size() * sizeof(gpu_slots.at(0))));
    LOG(INFO) << "COPYING GPU REQ " <<  gpu_slots.size() * sizeof(gpu_slots.at(0)) << " B";
    CUDA_CALL(cudaMemcpy(gpu_requests, gpu_slots.data(),
                         gpu_slots.size() * sizeof(gpu_slots.at(0)),
                         cudaMemcpyHostToDevice));
    if (mode != "gpu-gpu")
    {
        cpu_requests = (request_t **) cpu_slots.data();

        // CUDA_CALL(cudaMalloc(&cpu_requests, cpu_slots.size() * sizeof(gpu_slots.at(0))));
        // CUDA_CALL(cudaMemcpy(cpu_requests, cpu_slots.data(),
        //                      cpu_slots.size() * sizeof(cpu_slots.at(0)),
        //                      cudaMemcpyHostToDevice));
        // CUDA_CALL(cudaDeviceSynchronize());
    }
    LOG(INFO) << "cpu_requests is at " << cpu_requests;
    LOG(INFO) << "gpu_requests is at " << gpu_requests;

    executor->data = data;
    executor->batch = batch;
    executor->requests = gpu_requests;
    executor->cpu_requests = cpu_requests;
    // This is beacause the gpu addresses won't be directly accessible from the CPU
    if (mode != "gpu-gpu")
        executor->gpu_requests_cpu = (request_t **) gpu_slots.data();
    executor->n_requests = these_slots.size();
    executor->input_size = input_size;
    executor->output_size = output_size;
    executor->slot_size = slot_size;
    executor->max_inferences = max_inferences;
    if ((mode == "gpu-gpu") || (mode == "cpu-gpu"))
        executor->lkey = mr->lkey;
    else
        executor->lkey = cpu_mr->lkey;

    signal(SIGINT, on_exit_signal);
    auto killer_thread = self_killer();
    killer_thread_ptr = &killer_thread;


    CUDA_CALL(cudaDeviceSynchronize());
    if (!preload) {
        load_model(model_path, executor, async_metrics ? metrics_size : 0,
                   metrics_interval);
        warmup_model(executor, warmup_rounds);
    }


    // auto cupti_thread = cupti_metrics_thread(&stop);


    if (mode == "gpu-gpu" && profile > 1)
    {
        LOG(INFO) << "Profiling with the concurrent, pure, profiler";
        //instantiate_model_pp(executor, max_inferences, profile_concurrency, profile > 2, profile > 3);
        LOG(FATAL) << "Parallel profiler is disabled (for now!).";
        exit(1);

    } else if (mode == "gpu-gpu" || profile)
    {
        LOG(INFO) << "Launching magic tricks: bidirectional GPU mode";
        if (async_metrics)
            LOG(INFO) << "Using async metrics: it's normal to have no output until "
                         "CTRL+C!";
        if (profile) sleep_start = 0;
        executor->stop_on_finish = true;
        printf("EVENT WORKER_START\n");
        printf("RESULT-WORKER_START %li\n", now());
        instantiate_model(executor, true, sleep_start, profile, max_inferences, copy_mode);
        if (poll) {
            CUDA_CALL(cudaDeviceSynchronize());
        } else {
            while (!stop && !*executor->stop) {
                LOG(INFO) << "Still waiting...";
                sleep(1);
            }
        }
    }else if (mode == "cpu-gpu")
    {
        LOG(INFO) << "Launching magic tricks: RDMA-to-CPU input, RDMA-from-GPU output";
        if (async_metrics)
            LOG(INFO) << "Using async metrics: it's normal to have no output until "
                         "CTRL+C!";
        executor->stop_on_finish = true;
        printf("EVENT WORKER_START\n");
        printf("RESULT-WORKER_START %li\n", now());
        instantiate_model_cpu(executor, sleep_start, true, false, copy_io, copy_mode);
        CUDA_CALL(cudaDeviceSynchronize());

    }else if (mode == "cpu-cpu")
    {

        LOG(INFO) << "Launching magic tricks: RDMA-to-CPU input, RDMA-from-CPU output";
        if (async_metrics)
            LOG(INFO) << "Using async metrics: it's normal to have no output until "
                         "CTRL+C!";
        executor->stop_on_finish = true;
        printf("EVENT WORKER_START\n");
        printf("RESULT-WORKER_START %li\n", now());
        instantiate_model_cpu(executor, sleep_start, true, true, copy_io, copy_mode);
        CUDA_CALL(cudaDeviceSynchronize());



    }else if (mode == "gpu-cpu")
    {
        LOG(INFO) << "Launching magic tricks: RDMA-to-GPU input, RDMA-from-CPU output";
        LOG(FATAL) << "Not implemented yet. (Please implement me I'm feeling discriminated!)";
    } else {
        LOG(FATAL) << "Error: mode " << mode << " not implemented!";
    }


    printf("EVENT WORKER_STOP\n");
    printf("RESULT-WORKER_STOP %li\n", now());
    on_exit();
    // if (async_metrics) {
    //     LOG(INFO) << "Waiting for metrics...";
    //     if(!reported_metrics)
    //         {
    //             report_metrics(executor);
    //             reported_metrics = true;
    //         }
    // }


    return 0;
}
