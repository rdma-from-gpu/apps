#ifndef GRPC_H
#define GRPC_H

#include "rdma_from_gpu.pb.h"
extern "C" {
#include <rdma_shim.h>
}
#include "rdma_from_gpu.grpc.pb.h"
#include <string>
#include <grpc++/grpc++.h>
#include <future>
#include <infiniband/verbs.h>

using namespace rdma_from_gpu::proto;


struct ClientData{
    std::string peer;
    ClientHello client_hello;
    std::promise<void> worker_data_ready;

    WorkerHello worker_hello;
};


class GRPCWorker final : public RDMAServer::Service {
public:
    GRPCWorker(uint16_t port);
    void run_();
    std::thread spawn();

    grpc::Status Hello(::grpc::ServerContext *context, const ::rdma_from_gpu::proto::ClientHello *request,
                       ::rdma_from_gpu::proto::WorkerHello *response) override;

    // Just everything public... Don't learn from this!
    std::vector<ClientData *> clients_;

    std::unique_ptr<grpc::Server> server_;

    void shutdown();
};

WorkerHello grpc_client_connect(std::string remote_address,
                  std::string local_addr,
                   int qp_num,
                   ibv_mr * mr,
                   int slots, bool rc_transport = false);




#endif
