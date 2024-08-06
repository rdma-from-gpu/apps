#ifndef RDMA_FROM_GPU_BUFFERS_H
#define RDMA_FROM_GPU_BUFFERS_H

#include <stdint.h>

//#include "rdma.h"
#define CUDA_MEM_ALIGNMENT 64

// When a request is sent to the worker is written by-the-byte to the remote memory
// With the following data structure:
enum class request_status_t : uint8_t {UNUSED=0, INPUTS=1, OUTPUTS=2};
typedef struct __attribute((aligned(CUDA_MEM_ALIGNMENT), packed)){
    uint64_t id;
    //uint64_t client_id;
    uint32_t client_rkey; // for now, let's put them directly here.
    uint64_t client_slot;
    request_status_t status;
    //uint8_t data[0]; //  THis simplifies access
    uint64_t start_handling; // Just a quick way to store a timestamp
    uint64_t end_handling; // Just a quick way to store a timestamp
} request_t;


//int request_size(int input_size, int output_size)
//{
//    // TODO: Is this the correct alignment? Can we do better?
//    return sizeof(request_t) +  aligned_size_32(input_size) + aligned_size_32(output_size);
//}
//
//void * get_request_input(void * req)
//{
//    return (void*)  ((uint64_t) req + sizeof(request_t));
//}
//void * get_request_output(void * req, int input_size)
//{
//    return (void*)  ((uint64_t) req + sizeof(request_t) + aligned_size_32(input_size));
//}

#define request_size(input,output) (sizeof(request_t) +  aligned_size_32(input) + aligned_size_32(output))
//#define get_request_input(req) ((void *) ((uint64_t) req + sizeof(request_t)))
//#define get_request_output(req,input) ((void *) ((uint64_t) req + sizeof(request_t) + aligned_size_32(input)))
#define request_to_input(req) ((void*) ((uint64_t) req + sizeof(request_t)))

// This should work on 32-bits aligned sizes
#define request_to_output(req, input_size) ((void*) ((uint64_t) req + sizeof(request_t) + ((((( (uint64_t)input_size+ ( (1<<5) -1)) >> 5)) << 5))))



#endif //RDMA_FROM_GPU_BUFFERS_H
