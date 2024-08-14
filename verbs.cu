#include "include_all.h"
#include "rdma_shim.cuh"
#include "network.h"

int do_standard_write(ibv_qp *qp, int imm, void *buffer, int size, int lkey,
                      bool signaled, bool quiet) {
    if (buffer != 0 && size != 0)
        LOG_IF(INFO, !quiet) << "Standard write with imm. " << std::hex << imm;
    else
        LOG_IF(INFO, !quiet) << "Standard empty write with imm. " << std::hex << imm;
    return rdma_write_imm_standard(qp,
                                   (void *)buffer,     // buffer
                                   size,               // size
                                   lkey,               // lk
                                   0xbbaabbaa,         // rk
                                   (void *)0xffff1234, // raddr
                                   imm,                // imm
                                   signaled);
}


int do_gpu_crafted_write(struct rdma_shim_data *data, int imm, void *buffer,
                         int size, int lkey, bool signaled, bool quiet) {
    if (buffer != 0 && size != 0)
        LOG_IF(INFO, !quiet) << "GPU Crafted write with imm. " << std::hex << imm;
    else
        LOG_IF(INFO, !quiet) << "GPU Crafted empty write with imm. " << std::hex << imm;
    rdma_write_with_imm_kernel<<<1, 1, 1>>>(data,
                                            (void *)buffer,     // buffer
                                            size,               // size
                                            lkey,               // lk
                                            0xbbaabbaa,         // rk
                                            (void *)0xffff1234, // raddr
                                            imm,                // imm
                                            signaled);
    return 0;
}
int do_crafted_write(struct rdma_shim_data *data, int imm, void *buffer,
                     int size, int lkey, bool signaled, bool quiet) {
    if (buffer != 0 && size != 0)
        LOG_IF(INFO, !quiet) << "Crafted write with imm. " << std::hex << imm;
    else
        LOG_IF(INFO, !quiet) << "Crafted empty write with imm. " << std::hex << imm;
    rdma_write_with_imm_cu(data,
                           (void *)buffer,     // buffer
                           size,               // size
                           lkey,               // lk
                           0xbbaabbaa,         // rk
                           (void *)0xffff1234, // raddr
                           imm,                // imm
                           signaled);
    return 0;
}


