#include "grpc.h"
#include "rdma_from_gpu.grpc.pb.h"
#include <glog/logging.h>

using namespace grpc;


WorkerHello grpc_client_connect(std::string remote_address,
                  std::string local_addr,
                   int qp_num,
                   ibv_mr * mr,
                   int slots,
                   bool rc)
{

    LOG(INFO) << "Connecting to " << remote_address;
    auto channel = grpc::CreateChannel(remote_address, grpc::InsecureChannelCredentials());
    std::unique_ptr<RDMAServer::Stub> client = RDMAServer::NewStub(channel);
    ClientHello req;
    req.set_address(local_addr);
    req.set_addr((uint64_t) mr->addr);
    req.set_rkey(mr->rkey);
    req.set_length(mr->length);
    req.set_qpnum(qp_num);
    req.set_slots(slots);
    req.set_rc(rc);
    rdma_from_gpu::proto::WorkerHello reply;
    grpc::ClientContext context;

    Status s = client->Hello(&context, req, &reply);
    CHECK(s.ok()) << "Error while requesting: " << s.error_details();



    return reply;
}
