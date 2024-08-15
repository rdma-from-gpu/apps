#include "buffers.h"
#include "executor.h"
#include "include_all.h"
#include "network.h"
#include "rdma_shim.cuh"
#include "tvm/runtime/container/map.h"
#include "tvm/runtime/packed_func.h"
#include <dlpack/dlpack.h>

namespace pt = boost::property_tree;

//#define EXECUTOR_GPU_PRINT(...) printf(__VA_ARGS__)
#define EXECUTOR_GPU_PRINT(...)                                                \
    while (0) {                                                                \
    }

template <typename T>
std::vector<T> pt_as_vector(pt::ptree const &pt,
                            pt::ptree::key_type const &key) {
    std::vector<T> r;
    for (auto &item : pt.get_child(key))
        r.push_back(item.second.get_value<T>());
    return r;
}

int tvmshape2int(tvm::runtime::ShapeTuple shape) {
    int size = 1;
    for (int i = 0; i < shape->size; i++)
        size *= shape->data[i];
    return size;
}
inline DLDataType str_to_DLDataType(const char *t) {
    if (strcmp(t, "FP32") == 0) return DLDataType{kDLFloat, 32, 1};
    if (strcmp(t, "FP16") == 0) return DLDataType{kDLFloat, 16, 1};

    CHECK(0) << "Unsupported data type";
    return DLDataType{kDLFloat, 16, 1};
}

__global__ void wait_inputs_profile(executor_data *data) { data->next_input++; }

// This is a kernel responsible to "wait" for the inputs
__global__ void wait_inputs(executor_data *data) {
    // ... This should ideally be multi-thread
    // Each thread could look at a different value
    // And then we select the first it arrives (?)
    // But for now let's run it on a single thread.

    __syncthreads();
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        // We'll wait forever, looping until there is something ready or we get
        // the stop signal The "+1" would cause the very first inference to take
        // more time to execute Since we'll start to check from the second...
        // TODO: We should either implement current++ somewthere around the
        // output phase Or at least use a n_request that is 2^x so that it's
        // more efficient
        for (int j = (data->next_input + 1) % data->n_requests; !*data->stop;
             j = (j + 1) % data->n_requests) {
            request_status_t s = data->requests[j]->status;
            if (s == request_status_t::INPUTS) {
                data->next_input = j;
                data->requests[j]->start_handling = clock();
                break;
            }
            // printf("%i not ready\n", j);
        }
    }
    __syncthreads();
}

// This is a kernel to send to outputs to clients
__forceinline__ __device__ void _send_outputs(executor_data *data,
                                              request_t **requests) {
    // TODO: calculate the right offsets
    // __syncthreads();
    // if ((threadIdx.x==0) && (threadIdx.y == 0) && (threadIdx.z==0))
    // {
    request_t *request = requests[data->next_output];
    // void *local_output = request;
    void *local_output = request_to_output(request, data->input_size);
    // void *remote_output = (void*)request->client_slot;
    void *remote_output =
        request_to_output(request->client_slot, data->input_size);

    // get_request_output(requests[i]->client_slot, input_size);

    // Comment the following to loop over and over
    request->status = request_status_t::OUTPUTS;

    // printf("Sending a response: from %p to %p, %i B, lkey %i rkey %i\n",
    //         local_output, remote_output, data->output_size, data->lkey,
    //         request->client_rkey);

    rdma_write_with_imm_cu(data->data, local_output, data->output_size,
                           data->lkey, request->client_rkey, remote_output,
                           request->id,
                           (data->clocks->runs % data->batch) == 0);

    if ((data->clocks->runs % data->batch) == 0) {
        // printf("Consume cqe: %li mod %li\n",
        //         data->clocks->runs, data->batch);
        consume_cqe_cu(data->data);
    }

    // Let's tell the wait_inputs "check the next one"
    // Otherwise we would start looking at the same element the first round
    // data->current++;
    //}
    // Just for safety. Is it needed?
    __syncthreads();
    __threadfence_system();
}

// And this wraps the above as a proper kernel
__global__ void send_outputs(executor_data *data) {
    _send_outputs(data, data->requests);
}

__global__ void swap_pointers(executor_data *data) {
    // printf("(Should be) doing trickery with pointers for %i\n",
    // data->next_input);
    data->next_output = data->next_input;
    // TODO: Here we should change what is the actual model input/output
}

// These are helpers to copy the inputs and outputs to/from the GPU
__global__ void copy_inputs_memcpy(executor_data *data) {
    request_t *request = data->requests[data->next_input];
    void *input = request_to_input(request);
    // TODO: Is it more efficient to do a for loop and copy in parallel using
    // GPU clocks?
    memcpy(data->input, input, data->input_size);
    data->next_output = data->next_input;
}
__global__ void copy_outputs_memcpy(executor_data *data) {
    request_t *request = data->requests[data->next_output];
    void *output = request_to_output(request, data->input_size);
    // TODO: Is it more efficient to do a for loop and copy in parallel using
    // GPU clocks?
    memcpy(output, data->output, data->output_size);
}

__global__ void copy_inputs_kernel(executor_data *data) {
    request_t *request = data->requests[data->next_input];
    void *input = request_to_input(request);
    // memcpy(data->input, input,data->input_size);
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < (data->input_size / 8))
        ((uint64_t *)data->input)[k] = ((uint64_t *)input)[k];

    if (k == 0) data->next_output = data->next_input;
}
__global__ void copy_outputs_kernel(executor_data *data) {
    request_t *request = data->requests[data->next_output];
    void *output = request_to_output(request, data->input_size);
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < (data->output_size / 8))
        ((uint64_t *)data->output)[k] = ((uint64_t *)output)[k];
    // memcpy(output, data->output,data->output_size);
}

// These are all functions that are used to track the execution of the model,
// and the time components

__device__ uint64_t tvm_model_runs;

__global__ void inc_tvm_model_runs() { tvm_model_runs++; }

// This is for when we don't call looper. But we need to advance the counter
__global__ void inc_runs(executor_data *data) { data->clocks->runs++; }

__global__ void set_start_clock(clocks_t *clocks) {
    clocks->last_start = clock();
}
__global__ void add_clock(clocks_t *clocks) {
    // double t =
    //         clocks->sum /
    //          (1e-9 * clocks->runs * clocks->clock_hz);

    clocks->sum += (clock() - clocks->last_start);
    // printf("ADD_CLOCK %li -> %f \n", clocks->sum,t);
}
__global__ void set_this_start_clock(clocks_t *clocks) {
    clocks->last_start_this = clock();
}
__global__ void add_run_clock(clocks_t *clocks) {
    clocks->sum_run += (clock() - clocks->last_start_this);
}
// __global__ void add_run_clock2(executor_data * executor) {
//     double execution_time_graph_clock_run =
//             (executor->clocks->sum_run /
//              (1e-9 * executor->clocks->runs * executor->clocks->clock_hz));
//     printf("EXECUTION TIME %f for inference %li\n",
//     execution_time_graph_clock_run, executor->next_input);
//     executor->clocks->sum_run += (clock() -
//     executor->clocks->last_start_this);
// }
__global__ void add_wait_clock(clocks_t *clocks) {
    clocks->sum_wait += (clock() - clocks->last_start_this);
}
__global__ void add_send_clock(clocks_t *clocks) {
    clocks->sum_send += (clock() - clocks->last_start_this);
}
__global__ void add_total_handling(executor_data *executor) {
    auto r = executor->requests[executor->next_input];
    r->end_handling = clock();

    if (r->end_handling > r->start_handling) {
        executor->clocks->sum_total_handling +=
            r->end_handling - r->start_handling;
        executor->clocks->count_total_handling += 1;
    }
}
__global__ void add_copy_input_clock(clocks_t *clocks) {
    clocks->sum_copy_input += (clock() - clocks->last_start_this);
}
__global__ void add_copy_output_clock(clocks_t *clocks) {
    clocks->sum_copy_output += (clock() - clocks->last_start_this);
}

// This gets called periodically to store the metrics, which are later printed
__global__ void record_clocks(executor_data *executor) {
    if (((executor->clocks->runs % executor->metrics_interval) == 0 &&
         executor->clocks->runs != 0)) {
        // printf("record_clocks\n");
        if (executor->recorded_metrics >= executor->metrics_size) {
            printf("No more space for metrics!\n");
            return;
        }
        uint64_t runs = executor->clocks->runs - executor->clocks->last_runs;
        double execution_time_graph_clock =
            ((executor->clocks->sum / runs) * executor->clocks->tick_us);
        double execution_time_graph_clock_run =
            (executor->clocks->sum_run / runs) * executor->clocks->tick_us;
        double execution_time_graph_clock_wait =
            (executor->clocks->sum_wait / runs) * executor->clocks->tick_us;
        double execution_time_graph_clock_send =
            (executor->clocks->sum_send / runs) * executor->clocks->tick_us;

        double execution_time_graph_clock_copy_input =
            (executor->clocks->sum_copy_input / runs) *
            executor->clocks->tick_us;
        double execution_time_graph_clock_copy_output =
            (executor->clocks->sum_copy_output / runs) *
            executor->clocks->tick_us;

        double total_handling_clock = (executor->clocks->sum_total_handling /
                                       executor->clocks->count_total_handling) *
                                      executor->clocks->tick_us;

        uint64_t now = clock();
        // printf("throughput is runs %li * clock_hz %f / clock_diff %li\n",
        //         (executor->clocks->runs - executor->clocks->last_runs),
        //          executor->clocks->clock_hz,
        //         (now - executor->clocks->last_runs_clock));
        uint64_t elapsed = now - executor->clocks->last_runs_clock;
        double throughput = runs / (elapsed * executor->clocks->tick_us * 1e-9);

        metrics_t *m = &executor->metrics[executor->recorded_metrics];
        m->runs = executor->clocks->runs;
        m->wait_avg = execution_time_graph_clock_wait;
        m->run_avg = execution_time_graph_clock_run;
        m->send_avg = execution_time_graph_clock_send;
        m->copy_input_avg = execution_time_graph_clock_copy_input;
        m->copy_output_avg = execution_time_graph_clock_copy_output;
        m->total_avg = execution_time_graph_clock;
        m->total_handling_avg = total_handling_clock;
        // This should be accounted on the whole sum not on the single run,
        // hence *runs
        m->goodput = execution_time_graph_clock_run * runs / elapsed;
        // printf("GOODPUT: %f -> execution_time %f / elapsed %li\n",
        //         m->goodput,
        //         execution_time_graph_clock_run,
        //         elapsed);
        m->throughput = throughput;

        executor->clocks->sum_copy_input = 0;
        executor->clocks->sum_run = 0;
        executor->clocks->sum_copy_output = 0;
        executor->clocks->sum_total_handling = 0;
        executor->clocks->count_total_handling = 0;
        executor->clocks->sum_wait = 0;
        executor->clocks->sum_send = 0;
        executor->clocks->sum = 0;

        executor->clocks->last_runs_clock = now;
        executor->clocks->last_runs = executor->clocks->runs;
        executor->recorded_metrics++;
    } else if (executor->clocks->runs == 0) {
        // Initialize the first
        uint64_t now = clock();
        metrics_t *m = &executor->metrics[executor->recorded_metrics];
        m->runs = executor->clocks->runs;
        m->wait_avg = 0;
        m->run_avg = 0;
        m->send_avg = 0;
        m->total_avg = 0;
        m->goodput = 0;
        m->throughput = 0;

        executor->clocks->last_runs_clock = now;
        executor->clocks->last_runs = executor->clocks->runs;
        executor->recorded_metrics++;
    }
}

__global__ void print_clocks(executor_data *executor) {
    if ((executor->clocks->runs % executor->metrics_interval) == 0 &&
        executor->clocks->runs != 0) {
        double execution_time_graph_clock =
            (executor->clocks->sum /
             (executor->clocks->runs * executor->clocks->tick_us));
        double execution_time_graph_clock_run =
            (executor->clocks->sum_run /
             (executor->clocks->runs * executor->clocks->tick_us));
        double execution_time_graph_clock_wait =
            (executor->clocks->sum_wait /
             (executor->clocks->runs * executor->clocks->tick_us));
        double execution_time_graph_clock_send =
            (executor->clocks->sum_send /
             (executor->clocks->runs * executor->clocks->tick_us));
        double total_handling =
            (executor->clocks->sum_total_handling) /
            (executor->clocks->runs * executor->clocks->tick_us);

        uint64_t now = clock();
        double throughput =
            ((executor->clocks->runs - executor->clocks->last_runs) *
             executor->clocks->clock_hz) /
            (double)(now - executor->clocks->last_runs_clock);

        printf("GPU-%li-RESULT-GPU_WAIT_AVG  %f\n", executor->clocks->runs,
               execution_time_graph_clock_wait);
        printf("GPU-%li-RESULT-GPU_RUN_AVG  %f\n", executor->clocks->runs,
               execution_time_graph_clock_run);
        printf("GPU-%li-RESULT-GPU_SEND_AVG  %f\n", executor->clocks->runs,
               execution_time_graph_clock_send);
        printf("GPU-%li-RESULT-GPU_TOTAL_AVG  %f\n", executor->clocks->runs,
               execution_time_graph_clock);
        printf("GPU-%li-RESULT-GPU_THROUGHPUT %f\n", executor->clocks->runs,
               throughput);

        // How much is spent doing inference?
        printf("GPU-%li-RESULT-GPU_GOODPUT %f\n", executor->clocks->runs,
               execution_time_graph_clock_run / execution_time_graph_clock);

        printf("GPU-%li-RESULT-GPU_TOTAL_HANDLING %f\n", executor->clocks->runs,
               total_handling);

        executor->clocks->last_runs_clock = now;
        executor->clocks->last_runs = executor->clocks->runs;
    }
}

// This is a node that invokes the actual TVM graph
__global__ void looper(executor_data *data) {
    // TODO: The if branching is for sure detrimental
    auto g = cudaGetCurrentGraphExec();
    // printf("Looper: %li/%li\n", data->clocks->runs, limit);
    if (*data->stop) return;
    if ((data->max_inferences == 0) ||
        data->clocks->runs < data->max_inferences) {
        if (g) {
            data->clocks->runs++;
            int ret = cudaGraphLaunch(g, cudaStreamGraphTailLaunch);
        } else
            printf("Looper: graph is null\n");
    } else {
        printf("Looper: reached limits: %li/%li\n", data->clocks->runs,
               data->max_inferences);
        if (data->stop_on_finish) *data->stop = true;
    }
}

__global__ void launchTailGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
}

// Load the model from a .so file, then extract the pointers for easier use
void load_model(std::string folder, executor_data *executor,
                uint64_t metrics_size, uint64_t metrics_interval) {
    std::filesystem::path path = folder;
    auto so_file = path / "mod.so";
    auto json_file = path / "mod.json";
    auto params_file = path / "mod.params";
    auto metadata_file = path / "metadata.json";

    DLDevice dev{kDLCUDA, 0};
    tvm::runtime::Module mod_factory =
        tvm::runtime::Module::LoadFromFile(so_file);

    pt::ptree metadata;
    pt::json_parser::read_json(metadata_file, metadata);

    std::ifstream json_in(json_file.string().c_str(), std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)),
                          std::istreambuf_iterator<char>());
    json_in.close();

    auto create_graph_executor =
        tvm::runtime::Registry::Get("tvm.graph_executor.create");
    auto create_cuda_graph_executor =
        tvm::runtime::Registry::Get("tvm.graph_executor_cuda_graph.create");

    tvm::runtime::Module graph_executor = (*create_graph_executor)(
        json_data, mod_factory, (uint64_t)dev.device_type,
        (uint64_t)dev.device_id);

    executor->functions.cuda_graph_executor = (*create_cuda_graph_executor)(
        json_data, mod_factory, (uint64_t)dev.device_type,
        (uint64_t)dev.device_id);

    executor->functions.set_input =
        graph_executor.GetFunction("set_input_zero_copy");
    executor->functions.set_output =
        graph_executor.GetFunction("set_output_zero_copy");
    executor->functions.run = graph_executor.GetFunction("run");
    executor->functions.start_capture =
        executor->functions.cuda_graph_executor.GetFunction("start_capture");
    executor->functions.end_capture =
        executor->functions.cuda_graph_executor.GetFunction(
            "end_capture_device");
    executor->functions.run_cuda_graph =
        executor->functions.cuda_graph_executor.GetFunction("run_cuda_graph");
    executor->functions.get_cuda_graph =
        executor->functions.cuda_graph_executor.GetFunction("get_cuda_graph");
    executor->functions.get_cuda_graph_exec =
        executor->functions.cuda_graph_executor.GetFunction(
            "get_cuda_graph_exec");
    executor->functions.get_cuda_stream =
        executor->functions.cuda_graph_executor.GetFunction("get_cuda_stream");

    // WE should check all... But for sure if any of these are null
    // It is likely all are null
    CHECK(executor->functions.start_capture != nullptr)
        << "start_capture is null";
    CHECK(executor->functions.end_capture != nullptr) << "end_capture is null";
    CHECK(executor->functions.run_cuda_graph != nullptr)
        << "run_cuda_graph is null";
    executor->input_shape = pt_as_vector<long int>(metadata, "input_shape");
    executor->output_shape = pt_as_vector<long int>(metadata, "output_shape");

    executor->input_type =
        str_to_DLDataType(metadata.get<std::string>("input_type").c_str());
    executor->output_type =
        str_to_DLDataType(metadata.get<std::string>("output_type").c_str());

    executor->input_size =
        tvmshape2int(executor->input_shape) * executor->input_type.bits / 8;
    executor->output_size =
        tvmshape2int(executor->output_shape) * executor->output_type.bits / 8;

    CUDA_CALL(cudaMalloc(&executor->input, executor->input_size));
    CUDA_CALL(cudaMalloc(&executor->output, executor->output_size));

    executor->nd_input = tvm::runtime::NDArray::EmptyWrapper(
        executor->input_shape, executor->input_type, dev, executor->input);

    executor->nd_output = tvm::runtime::NDArray::EmptyWrapper(
        executor->output_shape, executor->output_type, dev, executor->output);

    LOG(INFO) << "input size is " << executor->input_size;
    LOG(INFO) << "output size is " << executor->output_size;

    executor->metrics_interval = metrics_interval;
    if (metrics_size) {
        CUDA_CALL(
            cudaMalloc(&executor->metrics, sizeof(metrics_t) * metrics_size));
        CUDA_CALL(
            cudaMemset(executor->metrics, 0, sizeof(metrics_t) * metrics_size));
        executor->cpu_metrics =
            (metrics_t *)malloc(sizeof(metrics_t) * metrics_size);
        memset(executor->cpu_metrics, 0, sizeof(metrics_t) * metrics_size);

        executor->metrics_size = metrics_size;
        executor->metrics_interval = metrics_interval;
        LOG(INFO) << "We'll record up to " << metrics_size * metrics_interval
                  << " execution";
    } else
        executor->metrics = nullptr;
}

// This is just to warmup the GPU and run the model a couple of time
// THis should also prevent JIT intervention after the first runs (if any)
void warmup_model(executor_data *executor, int warmup_rounds) {
    // TODO: We should hijack these functions as well...
    // But for now we just fill some dummy pointers there
    // int input_idx =
    // executor->functions.graph_executor.GetFunction("GetInputIdx")(executor->input_name)
    // TODO SUpport multiple inputs, and use input/output names
    int input_idx = 0;
    int output_idx = 0;
    executor->functions.set_input(input_idx, executor->nd_input);
    executor->functions.set_output(output_idx, executor->nd_output);

    // Just as warmup
    uint64_t start = now();
    for (uint64_t i = 0; i < warmup_rounds; i++)
        executor->functions.run();
    CUDA_CALL(cudaDeviceSynchronize());
    uint64_t stop = now();
    double execution_time_direct = (stop - start) / (warmup_rounds * 1.0);
    // LOG(INFO) << "A single execution (averaged on " << warmup_rounds << "
    // calls) should take " <<
    //     (stop - start) / (warmup_rounds * 1.0) << " ns";

    CUDA_CALL(cudaMalloc(&executor->clocks, sizeof(clocks_t)));
    CUDA_CALL(cudaMemset(executor->clocks, 0, sizeof(clocks_t)));

    auto max_inferences = executor->max_inferences;
    if (warmup_rounds > 0) {
        // Now we capture a run-only graph and run some times so that we can
        // estimate the times of a inference-only graph
        LOG(INFO) << "Starting graph capture";
        executor->functions.start_capture();
        auto tvm_stream =
            (cudaStream_t)(void *)executor->functions.get_cuda_stream();
        set_start_clock<<<1, 1, 0, tvm_stream>>>(executor->clocks);
        executor->functions.run();
        add_clock<<<1, 1, 0, tvm_stream>>>(executor->clocks);
        // if (executor->metrics)
        //     record_clocks<<<1, 1, 0, tvm_stream>>>(executor);
        // els
        //     print_clocks<<<1, 1, 0, tvm_stream>>>(executor);
        // Backup value
        executor->max_inferences = warmup_rounds;
        looper<<<1, 1, 1, tvm_stream>>>(executor);
        executor->functions.end_capture();
        CUDA_CALL(cudaDeviceSynchronize());
        cudaGraphExec_t tvm_run_graph_exec;
        auto tvm_run_graph =
            (cudaGraph_t)(void *)executor->functions.get_cuda_graph();
        CUDA_CALL(cudaGraphInstantiate(&tvm_run_graph_exec, tvm_run_graph,
                                       cudaGraphInstantiateFlagDeviceLaunch));
        DLOG(INFO) << "Graph instantiated at " << tvm_run_graph_exec;

        // executor->n_requests = warmup_rounds;
        start = now();
        CUDA_CALL(cudaGraphLaunch(tvm_run_graph_exec, tvm_stream));
        CUDA_CALL(cudaDeviceSynchronize());
        stop = now();
        // report_metrics(executor);

        double execution_time_graph = (stop - start) / (warmup_rounds * 1.0);
        // LOG(INFO) << "A single graph execution (averaged on " <<
        // warmup_rounds << " calls) should take " <<
        //     (stop - start) / (warmup_rounds * 1.0) << " ns";

        clocks_t clocks_cpu;
        CUDA_CALL(cudaMemcpy(&clocks_cpu, executor->clocks, sizeof(clocks_t),
                             cudaMemcpyDeviceToHost));

        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

        double execution_time_graph_clock =
            (clocks_cpu.sum / (1e-6 * clocks_cpu.runs * prop.clockRate));
        LOG(INFO) << "Expecting " << warmup_rounds << " found "
                  << clocks_cpu.runs;
        CHECK(clocks_cpu.runs == warmup_rounds) << " some rounds didn't run!";

        LOG(INFO) << "Execution statistics (out of " << clocks_cpu.runs
                  << " runs: "
                  << "direct " << execution_time_direct << " ns, "
                  << "graph " << execution_time_graph << " ns, "
                  << "graph clock " << execution_time_graph_clock << " ns";
    }

    // LOG(INFO) << "CUDA reports " << cpu_relaunches << " relaunches with "
    //     << cpu_cycles_sum << " gpu cycles, " << cpu_cycles_sum /
    //     (1.0*cpu_relaunches) << " each";
    // LOG(INFO) << "On a " << prop.clockRate <<  " kHz clock it is "
    //     << (cpu_cycles_sum / (1000.0 * cpu_relaunches * prop.clockRate)) << "
    //     s";
    executor->max_inferences = max_inferences;
}

// This is the actual setup of the model, where it gets loaded to the GPU and a
// Graph is built
void instantiate_model(executor_data *executor, bool run_it, int sleep_start,
                       bool profile, int profile_limit, int copy_mode) {
    CUDA_CALL(cudaMemset(executor->clocks, 0, sizeof(clocks_t)));
    CUDA_CALL(cudaDeviceSynchronize());

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

    LOG(INFO) << "clockRate " << prop.clockRate;
    // Assuming that it will be stable....
    double clock_hz = prop.clockRate * 1000.0;
    double tick_us = 1e9 / (prop.clockRate * 1000.0);
    CUDA_CALL(cudaMemcpy(&executor->clocks->clock_hz, &clock_hz,
                         sizeof(clock_hz), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(&executor->clocks->tick_us, &tick_us, sizeof(tick_us),
                         cudaMemcpyHostToDevice));

    int limit = profile ? profile_limit : 0;
    executor->max_inferences = limit;

    // Setup zero copy I/O
    int input_idx = 0;
    int output_idx = 0;
    executor->functions.set_input(input_idx, executor->nd_input);
    executor->functions.set_output(output_idx, executor->nd_output);

    // Capture an execution
    // We rely on TVM implementation for the capture
    // But thre is little secrete sauce there: it's a normal cuda graph capture
    LOG(INFO) << "Starting graph capture";
    executor->functions.start_capture();
    executor->tvm_stream =
        (cudaStream_t)(void *)executor->functions.get_cuda_stream();
    set_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    set_this_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    if (profile)
        wait_inputs_profile<<<1, 1, 1, executor->tvm_stream>>>(executor);
    else
        wait_inputs<<<1, 1, 1, executor->tvm_stream>>>(executor);

    add_wait_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    set_this_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    if (!profile) switch (copy_mode) {
        case 1:
            copy_inputs_memcpy<<<1, 1, 1, executor->tvm_stream>>>(executor);
            break;
        case 2: {
            int blocksize = 512; // TODO: Preliminary empyrical result
            int nblocks = (executor->input_size / blocksize) + 1;
            copy_inputs_kernel<<<blocksize, nblocks, 2048,
                                 executor->tvm_stream>>>(executor);
        }
        default:
            break;
        }
    add_copy_input_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    set_this_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    executor->functions.run();
    add_run_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    set_this_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    if (!profile) switch (copy_mode) {
        case 1:
            copy_outputs_memcpy<<<1, 1, 1, executor->tvm_stream>>>(executor);
            break;
        case 2: {
            int blocksize = 1024;
            int nblocks = (executor->output_size / blocksize) + 1;
            copy_outputs_kernel<<<blocksize, nblocks, 2048,
                                  executor->tvm_stream>>>(executor);
        }
        default:
            break;
        }
    add_copy_output_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    set_this_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    if (!profile) send_outputs<<<1, 1, 1, executor->tvm_stream>>>(executor);
    add_send_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    add_total_handling<<<1, 1, 0, executor->tvm_stream>>>(executor);
    add_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    if (executor->metrics)
        record_clocks<<<1, 1, 0, executor->tvm_stream>>>(executor);
    else
        print_clocks<<<1, 1, 0, executor->tvm_stream>>>(executor);
    looper<<<1, 1, 1, executor->tvm_stream>>>(executor);
    executor->functions.end_capture();
    CUDA_CALL(cudaDeviceSynchronize());

    // This is a "hacked" method so that we can get the actual cuda graph and
    // manipulate it here
    executor->tvm_graph =
        (cudaGraph_t)(void *)executor->functions.get_cuda_graph();

    // Here we need to pass "DeviceLaunch" to be able to run it from CUDA
    cudaGraphExec_t tvm_graph_exec;
    CUDA_CALL(cudaGraphInstantiate(&tvm_graph_exec, executor->tvm_graph,
                                   cudaGraphInstantiateFlagDeviceLaunch));
    LOG(INFO) << "Graph instantiated at " << tvm_graph_exec;

    // CUDA_CALL(cudaGraphDebugDotPrint(tvm_io_graph, "/tmp/tvm_graph.dot",
    // cudaGraphDebugDotFlagsVerbose |
    // cudaGraphDebugDotFlagsKernelNodeAttributes |
    // cudaGraphDebugDotFlagsHandles));

    cudaStream_t launcher_stream;
    cudaStreamCreate(&launcher_stream);

    // Setup some variables. Most aren't needed
    // This should also prevent the garbage collector to delete objects when we
    // don't execute directly
    executor->executor_stream = launcher_stream;
    executor->next_input = 0;

    if (run_it) {
        LOG(INFO) << "Waiting " << sleep_start
                  << " second before starting.. So that the client "
                     "can write";
        sleep(sleep_start);
        // Launch the host graph, which will in turn launch the device graph.
        // CUDA_CALL(cudaGraphLaunch(executor_graph_exec, launcher_stream));
        // int ret = cudaGraphLaunch(executor_graph_exec, launcher_stream);
        int ret = cudaGraphLaunch(tvm_graph_exec, launcher_stream);
        printf("cudaGraphLaunch returned %i\n", ret);
    }
}

cudaError_t run_executor(executor_data *executor) {
    LOG(INFO) << "Invoking executor with data at " << executor << ", for graph "
              << executor->executor_graph_exec << " on stream "
              << executor->executor_stream;
    return cudaGraphLaunch(executor->executor_graph_exec,
                           executor->executor_stream);
}

void report_metrics(executor_data *executor) {
    LOG(INFO) << "There are " << executor->recorded_metrics
              << " metrics to report";

    metrics_t *metrics_cpu =
        (metrics_t *)malloc(sizeof(metrics_t) * executor->metrics_size);
    CUDA_CALL(cudaMemcpy(metrics_cpu, executor->metrics,
                         sizeof(metrics_t) * executor->metrics_size,
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < executor->recorded_metrics; i++) {
        metrics_t *m = &metrics_cpu[i];
        metrics_t *cm = &executor->cpu_metrics[i];

        printf("GPU-%li-RESULT-GPU_RUN_AVG  %li\n", m->runs, m->run_avg);
        printf("GPU-%li-RESULT-GPU_TOTAL_AVG  %li\n", m->runs, m->total_avg);
        printf("GPU-%li-RESULT-GPU_THROUGHPUT  %f\n", m->runs, m->throughput);
        printf("GPU-%li-RESULT-GPU_GOODPUT  %f\n", m->runs, m->goodput);

        if (cm->wait_avg != 0)
            printf("GPU-%li-RESULT-GPU_WAIT_AVG  %li\n", cm->runs,
                   cm->wait_avg);
        else
            printf("GPU-%li-RESULT-GPU_WAIT_AVG  %li\n", m->runs, m->wait_avg);

        if (cm->total_handling_avg)
            printf("GPU-%li-RESULT-GPU_TOTAL_HANDLING_AVG  %li\n", cm->runs,
                   cm->total_handling_avg);
        else
            printf("GPU-%li-RESULT-GPU_TOTAL_HANDLING_AVG  %li\n", m->runs,
                   m->total_handling_avg);

        if (cm->send_avg != 0)
            printf("GPU-%li-RESULT-GPU_SEND_AVG  %li\n", cm->runs,
                   cm->send_avg);
        else
            printf("GPU-%li-RESULT-GPU_SEND_AVG  %li\n", m->runs, m->send_avg);

        printf("GPU-%li-RESULT-GPU_COPY_INPUT_AVG %li\n", m->runs,
               m->copy_input_avg);
        printf("GPU-%li-RESULT-GPU_COPY_OUTPUT_AVG %li\n", m->runs,
               m->copy_output_avg);

        printf("GPU-%li-RESULT-GPU_COPY_INPUT_CPU_AVG %li\n", m->runs,
               cm->copy_input_avg);
        printf("GPU-%li-RESULT-GPU_COPY_OUTPUT_CPU_AVG %li\n", m->runs,
               cm->copy_output_avg);
    }
}

// This is a wait inputs for the CPU cases
inline int wait_inputs_cpu(executor_data *data) {
    // We'll wait forever, looping until there is something ready or we get the
    // stop signal The "+1" would cause the very first inference to take more
    // time to execute Since we'll start to check from the second...
    uint64_t start_wait = now();
    int ret = 0;
    for (int j = (data->next_input + 1) % data->n_requests; !*data->stop;
         j = (j + 1) % data->n_requests) {
        request_status_t s = data->cpu_requests[j]->status;
        // printf("Looking at %li: %p\n", j, data->cpu_requests[j]);
        if (s == request_status_t::INPUTS) {
            // data->next_input = j;
            // We immediately change the status to avoid a 2nd processing
            // When we send, we'll set it to output, but potentially only
            // on GPU-side, so we want to update it also on CPU side.
            data->cpu_requests[j]->status = request_status_t::UNUSED;
            data->cpu_requests[j]->start_handling = now();
            ret = j;
            break;
        }
    }
    data->cpu_clocks->sum_wait += (now() - start_wait);
    return ret;
}

// A thread to send outputs so to not lock the main threads during processing of
// them
std::thread send_outputs_thread(executor_data *data, bool is_copy_needed) {
    return std::thread([=]() {
        int i;
        uint64_t start;
        cudaStream_t output_stream;
        CUDA_CALL(cudaStreamCreate(&output_stream));
        while (!*data->stop) {
            if (data->output_queue->wait_dequeue_timed(
                    i, std::chrono::milliseconds(1000))) {
                // LOG(INFO) << "Waiting for " << i << " to complete...";
                auto &output_event = data->output_events[i];
                CUDA_CALL(cudaStreamWaitEvent(output_stream, output_event, 0));
                // LOG(INFO) << "" << i << " completed";
                NVTX_PUSH_RANGE("send_outputs", 2)

                // TODO: here we should test both for CPU-side buffers and
                // GPU-side buffers. Since one may send reading from the GPU
                // memory directly.
                request_t *request = data->cpu_requests[i];
                request_t *gpu_request = data->gpu_requests_cpu[i];
                void *local_output =
                    request_to_output(request, data->input_size);
                void *remote_output =
                    request_to_output(request->client_slot, data->input_size);

                // We record always, so that we don't have this as overhead in
                // the copy case only
                CUDA_CALL(cudaEventRecord(data->start_copy_output_events[i],
                                          output_stream));
                // The following prevents g++ to reorder the instructions
                // (should not be needed, but better safe than nothing)
                asm volatile("" ::: "memory");

                if (is_copy_needed) {
                    CUDA_CALL(cudaMemcpyAsync(
                        request_to_output(request, data->input_size),
                        request_to_output(gpu_request, data->input_size),
                        data->output_size, cudaMemcpyDeviceToHost,
                        output_stream));
                }

                asm volatile("" ::: "memory");
                CUDA_CALL(cudaEventRecord(data->end_copy_output_events[i],
                                          output_stream));
                asm volatile("" ::: "memory");
                // CUDA_CALL(cudaStreamSynchronize(output_stream));
                CUDA_CALL(
                    cudaEventSynchronize(data->end_copy_output_events[i]));
                asm volatile("" ::: "memory");
                start = now();

                // We call the exact same routines, although we may eveb want to
                // compare to the standard libs
                rdma_write_with_imm_cu(
                    data->data, local_output, data->output_size, data->lkey,
                    request->client_rkey, remote_output, request->id,
                    (data->cpu_clocks->runs % data->batch) == 0);

                if ((data->cpu_clocks->runs % data->batch) == 0) {
                    // printf("Consume cqe: %li mod %li\n",
                    //         data->clocks->runs, data->batch);
                    consume_cqe_cu(data->data);
                }

                data->cpu_clocks->sum_send += (now() - start);
                request->end_handling = now();

                float input_ms, output_ms;
                CUDA_CALL(cudaEventElapsedTime(&input_ms,
                                               data->start_copy_input_events[i],
                                               data->input_events[i]));
                CUDA_CALL(cudaEventElapsedTime(
                    &output_ms, data->start_copy_output_events[i],
                    data->end_copy_output_events[i]));
                data->cpu_clocks->sum_copy_input += input_ms * 1e6;
                data->cpu_clocks->sum_copy_output += output_ms * 1e6;

                // Just a safety guard to avoid rollovers and overflows
                if (request->end_handling > request->start_handling &&
                    (request->end_handling - request->start_handling) <
                        10'000'000'000) {
                    data->cpu_clocks->sum_total_handling +=
                        (request->end_handling - request->start_handling);
                    data->cpu_clocks->count_total_handling += 1;
                }
                NVTX_POP_RANGE();
            }
        }
    });
}

void record_clocks_cpu(executor_data *executor) {
    if ((((executor->cpu_clocks->runs % executor->metrics_interval) == 0 &&
          executor->cpu_clocks->runs != 0))) {
        if (executor->recorded_cpu_metrics >= executor->metrics_size) {
            printf("No more space for metrics!\n");
            return;
        }

        uint64_t runs =
            executor->cpu_clocks->runs - executor->cpu_clocks->last_runs;
        double execution_time_graph_clock_wait =
            executor->cpu_clocks->sum_wait / runs;
        double execution_time_graph_clock_send =
            executor->cpu_clocks->sum_send / runs;

        double total_handling_clock =
            executor->cpu_clocks->count_total_handling
                ? (executor->cpu_clocks->sum_total_handling /
                   executor->cpu_clocks->count_total_handling)
                : 0;

        metrics_t *m = &executor->cpu_metrics[executor->recorded_metrics];
        m->runs = executor->cpu_clocks->runs;
        m->wait_avg = execution_time_graph_clock_wait;
        m->send_avg = execution_time_graph_clock_send;
        m->copy_input_avg = executor->cpu_clocks->sum_copy_input / runs;
        m->copy_output_avg = executor->cpu_clocks->sum_copy_output / runs;

        m->total_handling_avg = total_handling_clock;

        executor->cpu_clocks->sum_copy_input = 0;
        executor->cpu_clocks->sum_copy_output = 0;
        executor->cpu_clocks->sum_send = 0;
        executor->cpu_clocks->sum_wait = 0;
        executor->cpu_clocks->sum_total_handling = 0;
        executor->cpu_clocks->count_total_handling = 0;

        // executor->cpu_clocks->last_runs_clock = now;
        executor->cpu_clocks->last_runs = executor->cpu_clocks->runs;
        executor->recorded_cpu_metrics++;
    } else if (executor->clocks->runs == 0) {
        // Initialize the first
        uint64_t now = clock();
        metrics_t *m = &executor->cpu_metrics[executor->recorded_cpu_metrics];
        m->runs = executor->cpu_clocks->runs;
        m->wait_avg = 0;
        m->run_avg = 0;
        m->send_avg = 0;
        m->total_avg = 0;
        m->goodput = 0;
        m->throughput = 0;
        m->total_handling_avg = 0;

        // executor->cpu_clocks->last_runs_clock = now;
        // executor->cpu_clocks->last_runs = executor->clocks->runs;
        executor->recorded_cpu_metrics++;
    }
}

void instantiate_model_cpu(executor_data *executor, int sleep_start,
                           bool cpu_inputs, bool cpu_outputs,
                           bool is_copy_needed, int copy_mode) {
    CHECK(cpu_inputs || cpu_outputs)
        << "Either one of inputs or outputs should be cpu-mediated!";
    CUDA_CALL(cudaMemset(executor->clocks, 0, sizeof(clocks_t)));
    CUDA_CALL(cudaDeviceSynchronize());
    executor->cpu_clocks = (cpu_clocks_t *)malloc(sizeof(cpu_clocks_t));
    memset(executor->cpu_clocks, 0, sizeof(cpu_clocks_t));

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

    // Assuming that it will be stable....
    double clock_hz = prop.clockRate * 1000.0;
    double tick_us = 1e9 / (prop.clockRate * 1000.0);
    CUDA_CALL(cudaMemcpy(&executor->clocks->clock_hz, &clock_hz,
                         sizeof(clock_hz), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(&executor->clocks->tick_us, &tick_us, sizeof(tick_us),
                         cudaMemcpyHostToDevice));

    LOG(INFO) << "Launching a single inference...";
    executor->functions.run();
    CUDA_CALL(cudaDeviceSynchronize());

    // Capture an execution
    // We rely on TVM implementation for the capture
    // But thre is little secrete sauce there: it's a normal cuda graph capture
    // NOTE: No wait_inputs here since it will be the CPU launching the graph
    // every time!
    LOG(INFO) << "Starting graph capture";
    executor->functions.start_capture();
    executor->tvm_stream =
        (cudaStream_t)(void *)executor->functions.get_cuda_stream();
    set_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    // swap_pointers<<<1, 1, 1, executor->tvm_stream>>>(executor);

    switch (copy_mode) {
    case 0:
        swap_pointers<<<1, 1, 1, executor->tvm_stream>>>(executor);
        break;
    case 1:
    default:
        copy_inputs_memcpy<<<1, 1, 1, executor->tvm_stream>>>(executor);
        break;
    case 2:
        // TODO: FInd the best combinations
        int blocksize = 1024;
        int nblocks = (executor->input_size / blocksize) + 1;
        copy_inputs_kernel<<<nblocks, blocksize, 2048, executor->tvm_stream>>>(
            executor);
        break;
        sleep(1);
        exit(1);
    }
    add_copy_input_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);

    set_this_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    executor->functions.run();
    add_run_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    set_this_start_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    switch (copy_mode) {
    case 0:
        break;
    case 1:
    default:
        copy_outputs_memcpy<<<1, 1, 1, executor->tvm_stream>>>(executor);
        break;
    case 2:
        copy_outputs_kernel<<<1, 1, 1, executor->tvm_stream>>>(executor);
        break;
    }
    add_copy_output_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    if (!cpu_outputs) {
        send_outputs<<<1, 1, 1, executor->tvm_stream>>>(executor);
        add_send_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
        add_total_handling<<<1, 1, 0, executor->tvm_stream>>>(executor);
    }
    add_clock<<<1, 1, 0, executor->tvm_stream>>>(executor->clocks);
    if (executor->metrics)
        record_clocks<<<1, 1, 0, executor->tvm_stream>>>(executor);
    else
        print_clocks<<<1, 1, 0, executor->tvm_stream>>>(executor);
    // We need the following to advance the counters normally set by looper
    inc_runs<<<1, 1, 0, executor->tvm_stream>>>(executor);
    executor->functions.end_capture();
    CUDA_CALL(cudaDeviceSynchronize());

    // This is a "hacked" method so that we can get the actual cuda graph and
    // manipulate it here
    executor->tvm_graph =
        (cudaGraph_t)(void *)executor->functions.get_cuda_graph();

    // Here we need to pass "DeviceLaunch" to be able to run it from CUDA
    cudaGraphExec_t tvm_graph_exec;
    CUDA_CALL(cudaGraphInstantiate(&tvm_graph_exec, executor->tvm_graph, 0));
    // cudaGraphInstantiateFlagDeviceLaunch));
    LOG(INFO) << "Graph instantiated at " << tvm_graph_exec;

    cudaStream_t launcher_stream;
    cudaStreamCreate(&launcher_stream);

    // Setup some variables. Most aren't needed
    // This should also prevent the garbage collector to delete objects when we
    // don't execute directly
    executor->executor_stream = launcher_stream;
    executor->next_input = 0;

    LOG(INFO) << "Waiting " << sleep_start
              << " second before starting.. So that the client "
                 "can write";
    sleep(sleep_start);

    cudaStream_t input_stream;
    CUDA_CALL(cudaStreamCreate(&input_stream));

    // TODO: We don't need both everytime!
    executor->input_events =
        (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * executor->n_requests);

    std::thread output_thread;
    if (cpu_outputs) {
        executor->output_events =
            (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * executor->n_requests);
        executor->output_queue = new BlockingReaderWriterQueue<size_t>(128);
        output_thread = send_outputs_thread(executor, is_copy_needed);
    }

    executor->start_copy_input_events =
        (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * executor->n_requests);
    executor->start_copy_output_events =
        (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * executor->n_requests);
    executor->end_copy_output_events =
        (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * executor->n_requests);

    for (int i = 0; i < executor->n_requests; i++) {
        if (cpu_inputs) CUDA_CALL(cudaEventCreate(&executor->input_events[i]));
        if (cpu_outputs)
            CUDA_CALL(cudaEventCreate(&executor->output_events[i]));
        CUDA_CALL(cudaEventCreate(&executor->start_copy_input_events[i]));
        CUDA_CALL(cudaEventCreate(&executor->start_copy_output_events[i]));
        CUDA_CALL(cudaEventCreate(&executor->end_copy_output_events[i]));
    }
    // Just to be safe
    executor->cpu_clocks->sum_total_handling = 0;
    executor->cpu_clocks->count_total_handling = 0;

    CUDA_CALL(cudaDeviceSynchronize());

    while (!*executor->stop) {
        NVTX_PUSH_RANGE("Wait inputs", 1);
        int i = wait_inputs_cpu(executor);
        // executor->next_input = i;
        NVTX_POP_RANGE();

        auto &input_event = executor->input_events[i];
        // auto output_event = executor->events[i + executor->n_requests];
        // LOG_IF(INFO, is_copy_needed) << "Need to copy " <<
        // executor->cpu_requests[i]
        //     << " to " << executor->gpu_requests_cpu[i];

        // We record always, so that we don't have this as overhead in the copy
        // case only
        CUDA_CALL(cudaEventRecord(executor->start_copy_input_events[i],
                                  input_stream));

        if (is_copy_needed) {
            CUDA_CALL(cudaMemcpyAsync(
                // request_to_input(executor->gpu_requests_cpu[i]),
                // request_to_input(executor->cpu_requests[i]),
                executor->gpu_requests_cpu[i], executor->cpu_requests[i],
                executor->input_size +
                    sizeof(request_t), // We need to copy also the headers!
                cudaMemcpyHostToDevice, input_stream));
        } else {
            CHECK(executor->gpu_requests_cpu[i] == executor->cpu_requests[i])
                << "No copy should be needed... But your pointers don't match!";
        }

        // We use this so we are sure to not overwirte previous runs
        // Since this is forcerly synchornized with the finish of the previous
        // works
        CUDA_CALL(cudaMemcpyAsync(
            &executor->next_input, &i, sizeof(executor->next_input),
            cudaMemcpyHostToHost, executor->executor_stream));

        CUDA_CALL(cudaEventRecord(input_event, input_stream));
        CUDA_CALL(
            cudaStreamWaitEvent(executor->executor_stream, input_event, 0));
        cudaError_t ret =
            cudaGraphLaunch(tvm_graph_exec, executor->executor_stream);
        CHECK(ret == cudaSuccess) << "Error while launching graph...";
        // LOG(INFO) << "Launched processing for " << i;
        executor->cpu_clocks->runs++;
        if (cpu_outputs) {
            auto &output_event = executor->output_events[i];
            CUDA_CALL(cudaEventRecord(output_event, executor->executor_stream));
            executor->output_queue->enqueue(i);
        }

        if ((executor->cpu_clocks->runs % executor->metrics_interval) == 0) {
            record_clocks_cpu(executor);
        }
        // And that should be it. We don't need to explicitly synchronize:
        // it would "automatically" do it with the event, blocking until the
        // next copy has been done Which, in turn, would happen 1-at-time, and
        // consequently the execution
        // LOG(INFO)<< "Syncing";
        // CUDA_CALL(cudaDeviceSynchronize());
        // usleep(200);
    }
    LOG(INFO) << "End of the games... Stop is " << *executor->stop;
    if (cpu_outputs) output_thread.join();
}
