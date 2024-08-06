#include "grpc.h"
#include "include_all.h"



grpc::Status GRPCWorker::Hello(::grpc::ServerContext *context, const ::rdma_from_gpu::proto::ClientHello *request,
                               ::rdma_from_gpu::proto::WorkerHello *response) {
    LOG(INFO) << "New client: " << context->peer();
    ClientData * c = new ClientData();
    //c->peer = std::string(context->peer());
    c->client_hello = *request;
    auto f = c->worker_data_ready.get_future();
    clients_.push_back(c);
    f.get(); // This would wait for the main worker process to set things up
    LOG(INFO) << "Notifying client...";

    // We want to keep the original worker hello msg. + it's already allocated for the reply...
    response->set_qpnum(c->worker_hello.qpnum());
    response->set_addr(c->worker_hello.addr());
    response->set_rkey(c->worker_hello.rkey());
    response->set_length(c->worker_hello.length());
    response->set_slot_size((c->worker_hello.slot_size()));
    response->set_input_size((c->worker_hello.input_size()));
    response->set_output_size((c->worker_hello.output_size()));


    // Actually we could avoid this copy. But it's done just once...
    response->mutable_slots()->Add(c->worker_hello.mutable_slots()->begin(), c->worker_hello.mutable_slots()->end());

    return grpc::Status::OK;
}


GRPCWorker::GRPCWorker(uint16_t port)
{
  std::string server_address("0.0.0.0:");
  server_address+= std::to_string(port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
 builder.RegisterService(this);
 server_ = builder.BuildAndStart();
 std::cout << "Server listening on " << server_address << std::endl;
}

void GRPCWorker::run_ ()
{
    server_->Wait();
}
std::thread GRPCWorker::spawn(){
    return std::thread(&GRPCWorker::run_, this);
}
void GRPCWorker::shutdown()
{
    server_->Shutdown();
}
