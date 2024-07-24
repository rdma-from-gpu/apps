#include "include_all.h"
#include "rdma_shim.cuh"
#include "network.h"

void poll_cq_standard(ibv_cq *cq) {
    struct ibv_wc wc[1];
    int num_comp = ibv_poll_cq(cq, 1, wc);
    // LOG(INFO) << "Polled " << num_comp;
}

int poll_single_cq_standard(ibv_cq *cq, ibv_wc * wc) {
    int num_comp = ibv_poll_cq(cq, 1, wc);
    if (num_comp)
    {
        //LOG(INFO) << "Polled wc with id " << wc->wr_id;
    }
    else
    {
        //LOG(DEBUG) << "Nothing to poll";
    }
    return num_comp == 0;
}

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
