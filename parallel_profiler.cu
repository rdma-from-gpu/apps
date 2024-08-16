#include "buffers.h"
#include "executor.h"
#include "rdma_shim.cuh"
#include "rdma_shim.h"
#include "include_all.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <driver_types.h>
#include <thread>

static __device__ __inline__ uint64_t __nano() {
    uint64_t mclk;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
    return mclk;
}



__global__ void set_start_clock_pp(clocks_t *clocks) {
//    clocks->last_start = clock64();
    clocks->last_start = __nano();
}
__global__ void add_clock_pp(clocks_t *clocks, uint64_t * exec_times) {
    uint64_t t = (__nano() - clocks->last_start);
    clocks->sum += t;
    exec_times[clocks->runs] = t;
   // clocks->sum += (clock64() - clocks->last_start);
}

// This is a node that invokes the actual TVM graph
__global__ void looper_pp(executor_data *data, clocks_t * clocks) {
    // TODO: The if branching is for sure detrimental
    auto g = cudaGetCurrentGraphExec();
    // printf("Looper: %li/%li\n", data->clocks->runs, limit);
    if (data->stop && *data->stop) { printf("Got stop signal...\n"); return;}
    if ((data->max_inferences == 0) || clocks->runs <data->max_inferences) {
        if (g) {
                clocks->runs++;
                int ret = cudaGraphLaunch(g, cudaStreamGraphTailLaunch);
        } else
            printf("Looper: graph is null\n");
    } else
    {
        printf("Looper: reached limits: %li/%li\n", clocks->runs, data->max_inferences);
        if (data->stop_on_finish) *data->stop = true;
    }


}



void instantiate_model_pp(executor_data *executor, int profile_limit, int concurrency, bool all, bool cpu_launch) {
    CUDA_CALL(cudaMemset(executor->clocks, 0, sizeof(clocks_t)));
    CUDA_CALL(cudaDeviceSynchronize());

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

    LOG(INFO) << "clockRate " << prop.clockRate;
    // Assuming that it will be stable....
    double clock_hz = prop.clockRate * 1000.0;
    double tick_us = 1e9/(prop.clockRate * 1000.0);
    CUDA_CALL(cudaMemcpy(&executor->clocks->clock_hz, &clock_hz,
                         sizeof(clock_hz), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(&executor->clocks->tick_us, &tick_us,
                         sizeof(tick_us), cudaMemcpyHostToDevice));
    
    int max_inferences = ((profile_limit + concurrency - 1) / concurrency);
    LOG(INFO) << "Each graph would run " << max_inferences << " inferences";

    executor->max_inferences = max_inferences;

    // Capture an execution
    // We rely on TVM implementation for the capture
    // But thre is little secrete sauce there: it's a normal cuda graph capture
    std::vector<cudaGraphExec_t> tvm_execs;
    tvm_execs.reserve(concurrency);
    std::vector<cudaGraph_t> tvm_graphs;
    tvm_graphs.reserve(concurrency);
    std::vector<clocks_t*> clocks;
    clocks.reserve(concurrency);
    std::vector<cudaStream_t> streams;
    streams.reserve(concurrency);

    // This would hold **ALL** ecec times!
    std::vector<uint64_t*> exec_times;
    exec_times.reserve(concurrency);

    executor->functions.set_input("data", executor->nd_input);
    executor->functions.set_output("data", executor->nd_output);
    executor->functions.run();


    for (int i=0; i<concurrency; i++)
    {

        //TODO: SET INPUTS/OUTPUTS FOR EACH STREAM!

        CUDA_CALL(cudaMalloc((void**)&clocks[i], sizeof(clocks_t)));
        CUDA_CALL(cudaMalloc((void**)&exec_times[i], sizeof(uint64_t) * max_inferences));
        executor->functions.start_capture();
        executor->tvm_stream =
            (cudaStream_t)(void *)executor->functions.get_cuda_stream();
        set_start_clock_pp<<<1, 1, 0, executor->tvm_stream>>>(clocks[i]);
        executor->functions.run();
        add_clock_pp<<<1, 1, 0, executor->tvm_stream>>>(clocks[i], exec_times[i]);
        if (! cpu_launch)
            looper_pp<<<1, 1, 1, executor->tvm_stream>>>(executor, clocks[i]);
        executor->functions.end_capture();
        CUDA_CALL(cudaDeviceSynchronize());

        // This is a "hacked" method so that we can get the actual cuda graph and
        // manipulate it here
        LOG(INFO) << "Getting the graph";
        tvm_graphs[i] =
            (cudaGraph_t)(void *)executor->functions.get_cuda_graph();
        if (cpu_launch)
        {
            LOG(INFO) << "Instantiate a graph with CPU launching";
            CUDA_CALL(cudaGraphInstantiate(&tvm_execs[i], tvm_graphs[i],
                                           0));
        }
        else
        {
            CUDA_CALL(cudaGraphInstantiate(&tvm_execs[i], tvm_graphs[i],
                                           cudaGraphInstantiateFlagDeviceLaunch));
        }

        LOG(INFO) << "Graph instantiated at " << tvm_graphs[i];
        CUDA_CALL(cudaDeviceSynchronize());

        cudaStreamCreate(&streams[i]);
    }

    uint64_t start = now();
    if (cpu_launch)
    {
        LOG(WARNING) << "Using CPU launcher!";
        for (int j=0; j<max_inferences; j++)
        {
            for (int i=0; i< concurrency; i++)
            {
                int ret = cudaGraphLaunch(tvm_execs[i], streams[i]);
                // LOG(INFO) << "Launched graph " << i  << " with result " << ret;
            }
        }
    }
    else
    {
        for (int i=0; i<concurrency; i++)
        {
            int ret = cudaGraphLaunch(tvm_execs[i], streams[i]);
            LOG(INFO) << "Launched graph " << i  << " with result " << ret;
        }
    }
    CUDA_CALL(cudaDeviceSynchronize());
    uint64_t stop = now();
    LOG(INFO) << "Collecting metrics...";


    clocks_t cpu_clocks;
    uint64_t exec_time_avg = 0;
    uint64_t exec_count= 0;
    uint64_t* cpu_exec_times = (uint64_t *) malloc(sizeof(uint64_t) * max_inferences);

    std::ofstream times_file;
    if (all)
    {
        times_file.open ("times.txt", std::ios::out );
    }

    for(int i=0; i< concurrency; i++)
    {
        LOG(INFO)<<"Getting metrics for stream " << i;
        CUDA_CALL(cudaMemcpy(&cpu_clocks, clocks[i], sizeof(clocks_t), cudaMemcpyDeviceToHost));
        if (all){
            // LOG(INFO)<<"ALL " << i;
            CUDA_CALL(cudaMemcpy(cpu_exec_times, exec_times[i], sizeof(uint64_t) * max_inferences, cudaMemcpyDeviceToHost));
            for(int j=0; j<max_inferences; j++)
            {
                //LOG(INFO) << "exec times " << j << " for stream " << i << " " << cpu_exec_times[j];
                times_file << cpu_exec_times[j] << std::endl;
                //printf("RESULT-EXEC %li\n", cpu_exec_times[j]);
            }
        }

        double t= cpu_clocks.runs > 0 ?
            (cpu_clocks.sum /
             cpu_clocks.runs) : 0;
             // (1e-9 * cpu_clocks.runs * clock_hz));
        exec_time_avg+=t;
        exec_count += cpu_clocks.runs;
        printf("STREAM-%i-AVG_EXEC %f\n", i, t);
    }
    double rate = exec_count / ((stop-start)*1e-9);
    //printf("RESULT-AVG_RATE %f \n", (1.0*concurrency) / exec_time_avg);
    //printf("RESULT-AVG_RATE %f \n", exec_count / ((stop - start) * 1e-9));
    printf("RESULT-AVG_RATE %f \n", rate);
    // printf("RESULT-AVG_TIME %f \n", 1e9/rate);
    printf("RESULT-AVG_EXEC %f \n", (exec_time_avg)/ (1.0*concurrency));
    printf("RESULT-AVG_EXEC_MEAN %f \n", 1.0 / rate );

    if (all) 
        times_file.close();




}
