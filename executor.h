#ifndef RDMA_FROM_GPU_EXECUTOR_H
#define RDMA_FROM_GPU_EXECUTOR_H

// #include "rdma.h"
#include "rdma_shim.cuh"
#include "rdma_shim.h"
#include "buffers.h"
#include <thread>

#include "tvm/runtime/c_runtime_api.h"
#include <dlpack/dlpack.h>
#include <thread>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include "tvm/runtime/container/map.h"
#include "tvm/runtime/packed_func.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <readerwriterqueue.h>
using namespace moodycamel;
namespace pt = boost::property_tree;


typedef void (*WORKLOAD_T)(void*,int,void*,int);


// We define some macros to easily declare a launcher function and a kernel function
#define LAUNCH_EXECUTOR_NAME(name)  launch_simple_executor_##name
#define LAUNCH_EXECUTOR(name) \
    void LAUNCH_EXECUTOR_NAME(name) (int slot_size, int input_size, int output_size, \
                                     request_t ** requests, int n_requests,\
                                     rdma_shim_data * data,\
                                     uint32_t lkey,\
                                     bool loop,\
                                     int signaled_batch_size)

#define WORKLOAD_KERNEL(name) \
    __device__ void name##_kernel (void *inputs, int input_size, void* outputs, int output_size)

LAUNCH_EXECUTOR(t4_squeezenet_tuned);
WORKLOAD_KERNEL(t4_squeezenet_tuned);
LAUNCH_EXECUTOR(resnet50);
WORKLOAD_KERNEL(resnet50);
LAUNCH_EXECUTOR(forward);
WORKLOAD_KERNEL(forward);


struct clocks_t {
    uint64_t last_start;
    uint64_t last_start_this;
    uint64_t sum;
    uint64_t sum_run;
    uint64_t sum_wait;
    uint64_t sum_copy_input;
    uint64_t sum_copy_output;
    uint64_t sum_send;
    uint64_t sum_total_handling;
    uint64_t count_total_handling;
    uint64_t runs;
    double clock_hz;
    double tick_us;
    uint64_t last_runs;
    uint64_t last_runs_clock;
};

struct cpu_clocks_t {
    // uint64_t last_start;
    // uint64_t last_start_this;
    // uint64_t sum;
    // uint64_t sum_run;
    uint64_t sum_wait;
    uint64_t sum_send;
    uint64_t runs;
    uint64_t last_runs;
    uint64_t sum_total_handling;
    uint64_t count_total_handling;
    uint64_t sum_copy_input;
    uint64_t sum_copy_output;

    //uint64_t clock_hz;
    // uint64_t last_runs;
    // uint64_t last_runs_clock;
};


struct executor_functions {
    tvm::runtime::PackedFunc run;
    tvm::runtime::PackedFunc set_input;
    tvm::runtime::PackedFunc set_output;
    tvm::runtime::PackedFunc start_capture;
    tvm::runtime::PackedFunc end_capture;
    tvm::runtime::PackedFunc run_cuda_graph;
    tvm::runtime::PackedFunc get_cuda_graph;
    tvm::runtime::PackedFunc get_cuda_graph_exec;
    tvm::runtime::PackedFunc get_cuda_stream;
    tvm::runtime::Module cuda_graph_executor;
};
struct metrics_t {
    uint64_t runs;
    uint64_t wait_avg;
    uint64_t run_avg;
    uint64_t send_avg;
    uint64_t total_avg;
    uint64_t copy_input_avg;
    uint64_t copy_output_avg;
    uint64_t total_handling_avg;
    double throughput;
    double goodput;
};

struct executor_data {
    void * input; // TODO
    void * output; // TODO

    tvm::runtime::NDArray nd_input; // TODO
    tvm::runtime::NDArray nd_output; // TODO


    cudaGraph_t executor_graph;
    cudaGraphExec_t executor_graph_exec;
    cudaGraph_t tvm_graph;
    cudaGraphExec_t tvm_graph_exec;
    cudaGraph_t launcher_graph;
    cudaGraphExec_t launcher_graph_exec;

    cudaStream_t executor_stream;
    cudaStream_t tvm_stream;
    rdma_shim_data * data;
    int slot_size;
    int input_size;
    int output_size;
    request_t ** requests;
    int n_requests;
    uint32_t lkey;
    int batch;
    int next_input;
    int next_output;
    clocks_t * clocks;
    std::vector<long int> input_shape;
    std::vector<long int> output_shape;
    DLDataType input_type;
    DLDataType output_type;
    executor_functions functions;

    metrics_t * metrics;
    uint64_t metrics_size;
    uint64_t metrics_interval;
    uint64_t recorded_metrics;
    bool * stop;
    bool stop_on_finish;
    uint64_t max_inferences;


    // These are needed for the CPU mediated scenarios
    request_t ** cpu_requests;
    request_t ** gpu_requests_cpu;
    cudaEvent_t * input_events;
    cudaEvent_t * output_events;
    
    cudaEvent_t * start_copy_input_events;
    // cudaEvent_t * end_copy_input_events;
    cudaEvent_t * start_copy_output_events;
    cudaEvent_t * end_copy_output_events;



    //ReaderWriterQueue<size_t> input_queue(128);
    BlockingReaderWriterQueue<size_t> * output_queue;
    cpu_clocks_t * cpu_clocks;
    uint64_t recorded_cpu_metrics;
    metrics_t * cpu_metrics;

};


void load_model(std::string f, executor_data * executor, uint64_t metrics_size, uint64_t metrics_interval);
void warmup_model(executor_data * executor, int warmup_rounds);
void instantiate_model(executor_data * executor, bool run_it=true, int sleep_start = 3, bool profile = false, int profile_limit = 10000,  int copy_mode=1);
void instantiate_model_cpu(executor_data *executor, int sleep_start, bool cpu_inputs, bool cpu_outputs, bool is_copy_needed, int copy_mode=1);

cudaError_t run_executor(executor_data * executor);


// We'll wrap the following so that we have 2 distinguished functions
// for the cpu and gpu cases, but that don't have any conditional branching
// in them.
__forceinline__
__device__
void _send_outputs(executor_data * data, request_t ** requests);
__global__
void send_outputs(executor_data * data); //{_send_outputs(data, data->requests);}



__global__ void looper(executor_data *data);

std::thread poll_gpu_clocks(executor_data * executor, bool *stop);
void report_metrics(executor_data * executor);
void create_cpu_signaling(executor_data * data);
void instantiate_model_pp(executor_data * executor, int limit, int concurrency, bool all, bool cpu_launch);

std::thread cupti_metrics_thread(bool * stop);


#endif //RDMA_FROM_GPU_EXECUTOR_H
