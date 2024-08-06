#ifndef APPS_NETWORK_H
#define APPS_NETWORK_H

#include "include_all.h"
#define RDMA_NET_PATH "/sys/class/net"

int init_rdma(ibv_qp ** qp, ibv_pd ** pd, std::vector<std::string> ips, std::string remote_address, int dest_qp = 1,
              ibv_qp_attr ** qp_attr_ptr = nullptr, ibv_qp_type type = IBV_QPT_UC, bool single_cq=true);

std::pair<uint8_t, ibv_port_attr *> *get_dev_port(ibv_context *context,
                                                  ibv_device *dev);

struct ibv_gid_entry *
get_dev_gid(ibv_context *context, ibv_device *dev,
            std::pair<uint8_t, ibv_port_attr *> *port);

std::string get_dev_net_iface(ibv_device *device);

uint32_t get_dev_ip_addr(ibv_device *device);


ibv_device *get_device_by_ip(std::string address);

ibv_qp * create_qp(uint32_t max_send_wr, uint32_t max_recv_wr,
                                  uint32_t max_send_sge, uint32_t max_recv_sge,
								  ibv_cq *cq,
								  ibv_qp_type type,
								  ibv_pd * pd);
ibv_qp * create_qp(uint32_t max_send_wr, uint32_t max_recv_wr,
                   uint32_t max_send_sge, uint32_t max_recv_sge,
                   ibv_cq * send_cq,
                   ibv_cq * recv_cq,
                   ibv_qp_type type,
                   ibv_pd * pd);

ibv_cq * create_cq(int cqe, ibv_context * ctx, ibv_comp_channel * completion_channel);

void qp_to_rtr(ibv_qp *qp, ibv_mtu mtu, uint32_t dest_qp,
                               ibv_ah_attr attrs);
void qp_to_rts(ibv_qp *qp);

void qp_to_init(ibv_qp *qp, uint8_t port_num,
                                uint access_flags);

int finalize_rdma_connection(ibv_qp * qp, ibv_qp_attr * qp_attrs, int dest_qp);



void poll_cq_standard(ibv_cq *cq);
int poll_single_cq_standard(ibv_cq *cq, ibv_wc * wc);


int rdma_write_standard(ibv_mr * mr, ibv_qp * qp);
int rdma_send_standard(ibv_mr * mr, ibv_qp * qp);
int rdma_write_imm_standard(ibv_qp *qp, void * buffer, size_t size,
                                                uint32_t buffer_lkey,
        uint32_t buffer_rkey=0, void * raddr=0,
                                                int imm=0, bool signaled = false);



// Wrappers used in the generator
int do_standard_write(ibv_qp *qp, int imm, void *buffer, int size, int lkey,
                      bool signaled = false, bool quiet=true);


int do_crafted_write(struct rdma_shim_data *data, int imm, void *buffer,
                     int size, int lkey, bool signaled =false, bool quiet = true);

int do_gpu_crafted_write(struct rdma_shim_data *data, int imm, void *buffer,
                         int size, int lkey, bool signaled = false, bool quiet = true);
int post_recv(ibv_qp * qp);
#endif
