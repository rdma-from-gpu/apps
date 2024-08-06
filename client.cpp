#include <boost/program_options.hpp>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <numeric>
#include <unistd.h>
#include "grpc.h"
#include "buffers.h"
#include "network.h"

// Attention please! This would include also CUDA libs, although not needed at the client
// One should have a separate include_all file without cuda/rdma-shim libs, or a more
// streamline include are here...
#include "include_all.h"
#include <vector>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <future>
#include <csignal>
namespace po = boost::program_options;


struct client_request {
    request_t * local;
    request_t * remote;
    int id;
    uint64_t sent=0;
    uint64_t received=0;
    bool failed = false;
};
std::vector<client_request> requests;


std::string remote_address;
std::string local_address;
int remote_port;
bool quiet;
uint64_t buffer_size;
int batch;
//int gpu_batch;
//int gpu_buffer_size;
int n_slots;
int repeat;
std::atomic<uint64_t> outstanding = 0;
std::atomic<uint64_t> completed = 0;
std::atomic<uint64_t> timedout= 0;
int max_outstanding;
bool stop = false;
uint64_t total_requests = 0;
bool rc;
const int repeat_mult = 100000;
uint64_t timeout;
int max_runtime;
uint64_t start;

int arg_parse(int argc, char **argv, po::variables_map &vm) {
    // clang-format off
    po::options_description desc("RDMA from GPU: client");
            desc.add_options()
            ("help", "Print help message")
            ("address", po::value<std::string>(&remote_address)->default_value("127.0.0.1"), "Server address")
            ("local-address", po::value<std::string>(&local_address)->default_value("127.0.0.1"), "Server address")
            ("port", po::value<int>(&remote_port)->default_value(3333), "Server port")
            ("batch", po::value<int>(&batch)->default_value(2048), "How often to post signaled and consume cqe")
            //("write-size", po::value<int>(&write_size)->default_value(1000), "Buffer size to write")
            ("quiet", po::value<bool>(&quiet)->default_value(0), "Be less verbose")
            ("slots", po::value<int>(&n_slots)->default_value(1500), "How many slots to request")

            ("rc", po::value<bool>(&rc)->default_value(false), "Use RC transport")
            ("timeout", po::value<uint64_t>(&timeout)->default_value((uint64_t)5e9), "Timeout in ns")
            ("repeat", po::value<int>(&repeat)->default_value(1), "How many times to repeat")
            ("outstanding", po::value<int>(&max_outstanding)->default_value(0), "How many outstanding requests to have. 0 means n_slots")
            ("buffer-size", po::value<uint64_t>(&buffer_size)->default_value(1024*1024*1024), "Local buffer size")
            ("max-runtime", po::value<int>(&max_runtime)->default_value(260), "How long to run. Use it to avoid NPF errors!");
            //("remote-address", po::value<std::string>(&remote_address)->default_value("192.168.128.1"), "Remote address for the RDMA packets")
            //("gpu-batch", po::value<int>(&gpu_batch)->default_value(1), "Number of requests to send ayncrhonously when running from GPU.");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    // clang-format on

    if (vm.count("help")) {
        desc.print(std::cout);
        return 1;
    }
    if (max_outstanding == 0)
        max_outstanding = n_slots;

    return 0;
}

using namespace std;

int do_inference_write(
        ibv_qp * qp, ibv_mr * mr,
        int rkey,
        client_request * request,
        int input_size, int slot_size, bool signaled )
{
    //TODO: the header should be written as separate wr or sg to ensure ordering
    // Or at least put as "tail" so that we are almost sure it get written at the end

    
    request->local->client_slot = (uint64_t) request->local;
    request->local->client_rkey = (uint64_t) mr->rkey;
    request->local->status = request_status_t::INPUTS;
    request->local->id = request->id;

    ibv_send_wr * bad_wr;
    ibv_send_wr wr{};
    ibv_sge sglist{};

    memset(&wr, 0, sizeof(ibv_send_wr));
    memset(&sglist, 0, sizeof(ibv_sge));
    wr.num_sge=1;
    wr.sg_list = &sglist;

    sglist.addr = (uint64_t) request->local;
    DLOG(INFO) << "Sending local slot " << request->local << " with length " << slot_size << " to remote "<<request->remote;
    sglist.length = slot_size;
    sglist.lkey = mr->lkey;

    wr.wr.rdma.remote_addr = (uint64_t)request->remote;
    wr.wr.rdma.rkey = rkey;

    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.wr_id =request->id;

    wr.send_flags = signaled ? (IBV_SEND_SIGNALED | IBV_SEND_SOLICITED) : 0; // IBV_SEND_SOLICITED | IBV_SEND_FENCE;
    //wr.imm_data =;

    // printf("Sending a wr with sge at %p with %i wr\n",
    // wr.sg_list, wr.num_sge);
    request->sent = now();
    int ret = ibv_post_send(qp, &wr, &bad_wr);
    // printf("ibv_post_send returned %i\n", ret);
    return ret;
}

std::thread send_poller(ibv_qp * qp, bool *stop)
{
    // TODO: This is where we should check for the timings
    return std::thread ([qp, stop](){
        LOG(INFO) << "Starting send_poller thread";
        ibv_wc wc;
        while(!*stop){
            int ret = poll_single_cq_standard(qp->send_cq, &wc);
            if (!ret)
            {
                if (wc.opcode == IBV_WC_RDMA_WRITE)
                {
                    DLOG(INFO) << ibv_opcode_str(wc.opcode)
                              << " with id "<< wc.wr_id
                              << " and status " << ibv_wc_status_str(wc.status) ;
                    CHECK(wc.status == IBV_WC_SUCCESS) << "Error doing some RDMA magics";
                }
                else
                    LOG(INFO) <<  ibv_opcode_str(wc.opcode) << ": This event shouldn't be in the send cq..." <<
                    " Anyhow, its  status is " << ibv_wc_status_str(wc.status);
            }
        }
    });
}
std::thread recv_poller(ibv_qp * qp, bool *stop)
{
    // TODO: This is where we should check for the timings
    return std::thread ([qp, stop](){
        LOG(INFO) << "Starting recv_poller thread";
        ibv_wc wc;
        while(!*stop){
            int ret = poll_single_cq_standard(qp->recv_cq, &wc);
            if (!ret)
            {
                if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM)
                {
                    DLOG(INFO) << ibv_opcode_str(wc.opcode)
                            << " with id "<< wc.wr_id
                            << " with imm "<< wc.imm_data
                            << " and status " << ibv_wc_status_str(wc.status) ;
                    CHECK(wc.status == IBV_WC_SUCCESS) << "Error doing some RDMA magics";
                    outstanding--;
                    post_recv(qp);
                    if ((wc.imm_data) < requests.size())
                    {
                        requests[wc.imm_data].received = now();
                        // We receive only the actual outputs...
                        requests[wc.imm_data].local->status = request_status_t::OUTPUTS;
                        completed++;
                    }
                }
                else
                    LOG(INFO) <<  ibv_opcode_str(wc.opcode) << ": This event shouldn't be in the recv cq..." <<
                    " Anyhow, its  status is " << ibv_wc_status_str(wc.status);
            }
        }
    });
}

inline void print_status(bool on_exit = false)
{
        uint64_t t = now();
        printf("%li-RESULT-CLIENT_OUTSTANDING %li\n", t, outstanding.load());
        printf("%li-RESULT-CLIENT_COMPLETED %li\n", t, completed.load());
        printf("%li-RESULT-CLIENT_TIMEDOUT %li\n", t, timedout.load());
        printf("%li-RESULT-CLIENT_RATE %f\n", t, completed.load() / ( (t - start) / 1e9) );
        if(on_exit)
        {
        printf("RESULT-CLIENT_OUTSTANDING %li\n", outstanding.load());
        printf("RESULT-CLIENT_COMPLETED %li\n", completed.load());
        printf("RESULT-CLIENT_TIMEDOUT %li\n", timedout.load());
        printf("RESULT-CLIENT_RATE %f\n", completed.load() / ( (t - start) / 1e9) );
        }

    // std::string s = "Outstanding ("+ std::to_string(outstanding)+"): [ ";

    // for(auto &r:requests)
    //     if (r.received == 0 && r.sent != 0 && ! r.failed)
    //         s+= std::to_string(r.id) + ", ";
    //     else if (r.received == 0 && r.sent == 0)
    //         break;
    // s+= " ]";
    // return s;
}

inline void print_times(bool onexit = false)
{
    uint64_t sum = 0;
    uint64_t exclude = 0.1 * completed;
    std::string s = "Completion times: [ ";
    std::vector<uint64_t> times;
    times.reserve(requests.size());

    for(auto &r:requests)
    {
        if (exclude > 0)
        {
            // This would exclude the initial requests we did
            exclude--;
            continue;
        }
        // LOG(INFO) << r.id << "  " << r.received << " "<<r.sent;
        if (r.received != 0 && r.sent != 0 && !r.failed)
        {
            times.push_back(r.received - r.sent);
        }
        else
            if (r.received == 0 && r.sent == 0) // no point to go beyond this
             break;
    }
    for (auto &t : times)
        sum+=t;

    double avg = sum/(1.0*times.size());
    std::sort(times.begin(),times.end());
    // Dumb way...
    double a95 = 0;
    int tails_n = times.size() * (100-95) / 100 / 2;
    for (int i=tails_n; i< times.size() - tails_n; i++)
        a95+=times[i];
    a95/=(times.size() * 0.95);
    uint64_t t = now();
    printf("%li-RESULT-CLIENT_AVG %f\n",t, sum/(1.0*times.size()));
    printf("%li-RESULT-CLIENT_A95 %f\n",t, a95);
    if (onexit)
    {
        printf("RESULT-CLIENT_AVG %f\n", sum/(1.0*times.size()));
        printf("RESULT-CLIENT_A95 %f\n", a95);
    }

}

std::thread status_printer(bool *stop)
{
    // TODO: This is where we should check for the timings
    return std::thread ([requests,stop](){
        LOG(INFO) << "Starting status thread";
        while(!*stop)
        {
            usleep(500'000);
            print_status();
            print_times();
        }
     });
}

std::thread handle_timeouts(bool *stop)
{
    // TODO: This is where we should check for the timings
    return std::thread ([requests,stop](){
        LOG(INFO) << "Starting timeout thread";
        while(!*stop)
        {
            // TODO: we should handle thread-safeness somehow
            uint64_t max_old = now() - timeout;
            uint64_t timedout_this_round =0;
            for (int i=0; i< requests.size(); i++)
            {
                if (requests.at(i).sent == 0)
                    break;
                if (requests.at(i).sent != 0 && requests.at(i).received == 0 && requests.at(i).sent < max_old)
                    {
                        requests.at(i).received = max_old;
                        requests.at(i).failed = true;
                        //printf("request %i timedout\n", i);
                        timedout_this_round++;
                        outstanding--;
                        timedout++;
                    }
            }
        printf("%li requests timed out this round\n", timedout_this_round);

        usleep(1'000'000);
        }
     });

}


void on_exit(int signal)
{
    stop = 1;
    print_status(true);
    print_times(true);
}

std::thread self_killer(bool *stop)
{
    return std::thread([stop](){
            for(int i = 0; i< max_runtime && !*stop; i++)
                sleep(1);
            if (!*stop)
            {
                // This means we reached the end of the for loop
                LOG(INFO) << "Reached max runtime of " << max_runtime << "s.";
                raise(SIGINT);
            }
            });
}

int main(int argc, char **argv) {
    po::variables_map vm;
    int ret = arg_parse(argc, argv, vm);
    if (ret) return 1;



    ibv_qp * qp;
    ibv_pd * pd = nullptr;
    ibv_qp_attr * qp_attrs;
    std::vector<std::string> ips;
    ips.push_back(local_address);
    LOG(INFO) << local_address;
    CHECK(init_rdma(&qp, &pd, ips, remote_address, -1,
                    &qp_attrs,
                    rc ?IBV_QPT_RC : IBV_QPT_UC
                    , false) == 0) << "error while initializing RDMA stack!";

    void * buffer = malloc(buffer_size);
    CHECK(buffer) << "Error while allocating buffer";

    auto mr = ibv_reg_mr(qp->pd, buffer, buffer_size,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    CHECK(mr) << "Error while registering buffer";

    int fd = open("/dev/random", O_RDONLY);
    read(fd, buffer, buffer_size);
    close(fd);

    WorkerHello worker_hello = grpc_client_connect(remote_address + ":" + std::to_string(remote_port),
            local_address,
            qp->qp_num,
            mr, n_slots, rc);

    int dst_qp = worker_hello.qpnum();
    finalize_rdma_connection(qp, qp_attrs, dst_qp);
    LOG(INFO) << "Established connection with client " << remote_address << " and QP "<< qp->qp_num << " " << dst_qp;

    LOG(INFO) << "Got " << worker_hello.slots().size() << " slots";

    // // std::vector<void*> remote_slots;
    // std::vector<void*> local_slots;

    // // There must be another way!
    // remote_slots.reserve(worker_hello.slots().size());
    // local_slots.reserve(worker_hello.slots().size());

    // for(auto & s : worker_hello.slots())
    //     remote_slots.push_back( (void*) s);

    int received_slots = worker_hello.slots().size();
    int slot_size = worker_hello.slot_size();
    int client_id = worker_hello.client_id();
    int input_size = worker_hello.input_size();
    int rkey = worker_hello.rkey();
    LOG_IF(INFO, received_slots < n_slots) << "Warning! Received less slots than requested. Proceed anyway";
    CHECK(buffer_size > received_slots * slot_size) << "You should preallocate more memory! We need " << slot_size << "*"<< received_slots << " = " << slot_size * received_slots<< "B to handle so many slots!"; 
    LOG(INFO) << "Server rkey is " << rkey;

    requests.reserve(received_slots * repeat);
    LOG(INFO) << "received slots is  " << received_slots << " and repeat " << repeat;
    int id = 0;
    for (int r = 0; r < repeat; r++)
    {
        for(int i=0; i<received_slots; i++)
        {
            client_request c;

            // The local is always different, the remote repeats itself
            // c.local = (request_t *)((uint64_t)buffer + id * slot_size);
            c.local = (request_t *)((uint64_t)buffer + i * slot_size);
            c.remote = (request_t  *)worker_hello.slots().at(i);
            // c.id = (r*repeat_mult)+i;
            c.id = id;
            requests.push_back(c);
            id++;
        }
    }


    int n_recv = max(max_outstanding*4, received_slots);
    n_recv = n_recv > qp->recv_cq->cqe ? qp->recv_cq->cqe: n_recv;
    LOG(INFO) << "Filling recv queue with " << n_recv << " entries";

    // TODO: Check that this is not higher than queue capacity....
    for(int i=0; i<max_outstanding*2 && ret == 0 ; i++)
        ret = post_recv(qp);

    // Here we can start with the proper client things (= send reads)
    LOG(INFO) << "CQ are " << qp->send_cq << " " << qp->recv_cq;
    auto send_poller_th = send_poller(qp, &stop);
    auto recv_poller_th = recv_poller(qp, &stop);
    auto status_th = status_printer(&stop);
    auto timeouts_th = handle_timeouts(&stop);

   signal(SIGINT,on_exit);
   signal(SIGTERM,on_exit);
   auto killer_thread = self_killer(&stop);


    if (max_outstanding > received_slots)
    {
        LOG(INFO) << "Adjusting outstanding requests to " <<received_slots;
        max_outstanding = received_slots;
    }
    LOG(INFO) << "Max outstanding " << max_outstanding;
    LOG(INFO) << "received slots " << received_slots;

    start  = now();
    printf("EVENT CLIENT_START\n");
    printf("RESULT-CLIENT_START %li\n", start);

    int slow_start_limit = max_outstanding * 2;
    bool slow_start = false;
    int current_limit = 8;
    int last_increment = 0;
    for(auto &r : requests)
    {
        // Here we should check the outstanding requests, and prevent them to overwrite existing input data...
        // So we should look if the local request has received timestamp or not in the previous round.
        while(outstanding >= max_outstanding && !stop){
            usleep(10);
        }
        if (slow_start){
            while((outstanding >= current_limit) && !stop){
                usleep(10);
            }
            if ((completed - last_increment) > current_limit)
            {
                current_limit = current_limit * 2;
                last_increment = total_requests;
                LOG(INFO) << "Slow start limit increased to " << current_limit;
            }
            if (current_limit > slow_start_limit)
            {
                LOG(INFO) << "End of slow start phase!";
                slow_start = false;
            }
        }


            // We could even randomize local slots, but it shouldn't change performance worker-side
            do_inference_write(qp, mr,
                               rkey,
                               &r,
                               input_size, slot_size,
                               (total_requests%batch)== 0);
            //usleep(100);
            outstanding++;
            total_requests++;
            if (stop)
                break;
    }

    uint64_t max_waiting = 5'000'000'000;
    uint64_t start_waiting = now();
    while(outstanding >0 && (now() - start_waiting) < max_waiting && !stop)
    {
        LOG(INFO) << "Waiting for completions... " << outstanding << " missing";
        usleep(100000);
    }
    uint64_t t_stop  = now();
    LOG_IF(INFO, outstanding>0) << "Giving up: there are still some missing results!";
    printf("EVENT CLIENT_STOP\n");
    printf("RESULT-CLIENT_STOP %li\n", now());

    LOG(INFO) << "Done " << completed << " requests in " <<  (t_stop - start) / 1e9 << " s: " << ((t_stop - start) / 1e6) / completed << " ms/req";

    on_exit(0);
    // JUst ignore all proper handling 
    exit(0);

    send_poller_th.join();
    killer_thread.join();
    recv_poller_th.join();
    status_th.join();
    timeouts_th.join();


    LOG(INFO) << "Hejdȧ";

    exit(0);
    return 0;
}
