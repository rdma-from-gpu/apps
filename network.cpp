#include "network.h"
#include "include_all.h"
#include <rdma_shim.h>


int init_rdma(ibv_qp ** qp, ibv_pd ** pd, std::vector<std::string> ips, std::string remote_address, int dest_qp,
              ibv_qp_attr ** qp_attrs_ptr,
              ibv_qp_type type,
              bool single_cq)
{
    ibv_device *dev;
    std::string local_address;
    for (int i = 0; i < ips.size(); i++) {
        dev = get_device_by_ip(ips[i]);
        if (dev) {
            local_address = ips[i];
            LOG(INFO) << "Local IP is " << local_address;
            break;
        }
    }
    auto ctx = ibv_open_device(dev);
    if (*pd == nullptr)
         *pd = ibv_alloc_pd(ctx);

    int max_wr = 1024;
    int max_sge = 8;
    int cqe = 4095;
    auto completion_channel = ibv_create_comp_channel(ctx);
    auto send_cq = create_cq(cqe, ctx, completion_channel);
    auto recv_cq = single_cq ? send_cq : create_cq(cqe, ctx, completion_channel);

    *qp = create_qp(max_wr, max_wr, max_sge, max_sge, send_cq, recv_cq, type, *pd);
    CHECK(qp) << "Error while creating QP!";

    auto port_info = get_dev_port(ctx, dev);
    if (port_info->second->state != IBV_PORT_ACTIVE) {
        LOG(ERROR) << "Device " << dev->name << " is not active, aborting.";
        return 1;
    }

    auto gid_entry = get_dev_gid(ctx, dev, port_info);

    DLOG(INFO) << "Initializing QP";
    qp_to_init(*qp, port_info->first,
               IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    auto *remote_gid = new union ibv_gid;
    memcpy(remote_gid, gid_entry, sizeof(union ibv_gid));
    uint32_t remote_addr;
    inet_pton(AF_INET, remote_address.c_str(), &remote_addr);
    remote_gid->raw[12] = remote_addr & 0x000000ff;
    remote_gid->raw[13] = (remote_addr & 0x0000ff00) >> 8;
    remote_gid->raw[14] = (remote_addr & 0x00ff0000) >> 16;
    remote_gid->raw[15] = (remote_addr & 0xff000000) >> 24;

    auto *qp_attrs = new ibv_qp_attr;
    memset(qp_attrs, 0, sizeof(ibv_qp_attr));
    qp_attrs->ah_attr.port_num = port_info->first;
    qp_attrs->ah_attr.grh =
        (ibv_global_route){.dgid = *remote_gid,
                           .flow_label = 0x0,
                           .sgid_index = gid_entry->gid_index,
                           .hop_limit = 64,
                           .traffic_class = 0x0};
    qp_attrs->ah_attr.is_global = 1;
    (*qp)->qp_context = reinterpret_cast<void *>(qp_attrs);
    // Do we need the followings?
    (*qp)->send_cq = send_cq;
    (*qp)->recv_cq = recv_cq;

    LOG(INFO) << "dest_qp is " << dest_qp;
    if (dest_qp > -1)
        finalize_rdma_connection(*qp, qp_attrs, dest_qp);

    if (qp_attrs_ptr!= nullptr)
        *qp_attrs_ptr = qp_attrs;

    return 0;
}


ibv_device *get_device_by_ip(std::string address) {
    int num_devices = 0;
    ibv_device **devs = ibv_get_device_list(&num_devices);

	LOG(INFO) << "There are " << num_devices << "devices";
    if (!devs || num_devices == 0) return nullptr;

    uint32_t target_addr;
    inet_pton(AF_INET, address.c_str(), &target_addr);
    ibv_device *dev = nullptr;
    for (uint32_t i = 0; devs[i] && i < num_devices; ++i) {
        uint32_t dev_ip_addr = get_dev_ip_addr(devs[i]);
        LOG(INFO) << "DEVICE IP:" << dev_ip_addr << " TARGET " <<
        target_addr;

        if (dev_ip_addr > 0 && target_addr == dev_ip_addr) {
            dev = devs[i];
            LOG(INFO) << "Found ib_device for IP " << address << ": "
                      << dev->name;

            break;
        }
    }

    if (!dev) return nullptr;

    ibv_free_device_list(devs);

    return dev;
}

std::pair<uint8_t, ibv_port_attr *> *get_dev_port(ibv_context *context,
                                                  ibv_device *dev) {
    std::string net_dev = get_dev_net_iface(dev);
    if (net_dev.empty()) return nullptr;

    uint8_t port = 0;
    uint8_t port1 = 0;
    auto net_path = std::filesystem::path{RDMA_NET_PATH};
    std::ifstream dev_id_stream(net_path / net_dev / "dev_id");
    if (dev_id_stream.good()) {
        port1 = 0;
        std::ifstream dev_port_stream(net_path / net_dev / "dev_port");
        if (dev_port_stream.good()) {
            std::string dev_port;
            getline(dev_port_stream, dev_port);
            port1 = std::stoi(dev_port);
        }
        dev_port_stream.close();

        std::string dev_id;
        getline(dev_id_stream, dev_id);
        unsigned int port64;
        sscanf(dev_id.c_str(), "%x", &port64);
        port64 = port;
    } else {
        LOG(ERROR) << "Could not find port for device " << net_dev;
    }
    dev_id_stream.close();

    port = ((port1 > port) ? port1 : port) + 1;

    LOG(INFO) << "Device " << dev->name << " port is " << (int)port;

    auto *port_attr = new ibv_port_attr;
    ibv_query_port(context, port, port_attr);

    return new std::pair<uint8_t, ibv_port_attr *>(port, port_attr);
}

struct ibv_gid_entry *
get_dev_gid(ibv_context *context, ibv_device *dev,
            std::pair<uint8_t, ibv_port_attr *> *port) {
    uint32_t ip_addr = get_dev_ip_addr(dev);

    for (uint8_t gid_num = 0; gid_num < port->second->gid_tbl_len - 1;
         ++gid_num) {
        auto entry = new ibv_gid_entry();
        ibv_query_gid_ex(context, port->first, gid_num, entry,0);
        if (entry->gid_type != ibv_gid_type::IBV_GID_TYPE_ROCE_V2)
            continue;

        if (entry->gid.raw[0] == 0xfe && entry->gid.raw[1] == 0x80) // IPv6, skip
            continue;

        if (entry->gid.raw[10] == 0xff &&
            entry->gid.raw[11] == 0xff) { // IPv4, check if they match
            uint32_t gid_addr = entry->gid.raw[12] | entry->gid.raw[13] << 8 |
                                entry->gid.raw[14] << 16 | entry->gid.raw[15] << 24;

            if (gid_addr == ip_addr) {
                LOG(INFO) << "GID of device " << context->device->name << " is "
                          << (int)gid_num;
                    return entry;

            }
        }

        delete entry;
    }

    return nullptr;
}

std::string get_dev_net_iface(ibv_device *device) {
    /* Read IB device resource identifier from
     * /sys/class/net/{iface}/device/resource */
    std::string ib_res;
    std::ifstream ib_res_stream(std::string(device->dev_path) +
                                "/device/resource");
    while (!ib_res_stream.eof()) {
        std::string data;
        ib_res_stream >> data;
        ib_res += data;
    }
    ib_res_stream.close();

    std::string net_iface = "";

    std::filesystem::path net_dir{RDMA_NET_PATH};
    for (auto const &dir : std::filesystem::directory_iterator{net_dir}) {
        if (dir.path().filename() != "lo") {
            std::string curr_iface = dir.path().filename();
            // DLOG(INFO) << "Look at interface " << dir.path();

            /* Read net device resource identifier from
             * /sys/class/net/{iface}/device/resource */
            std::string net_res;
            auto res = std::filesystem::path{"device/resource"};
            std::ifstream net_res_stream(dir / res);
            if (!net_res_stream.good()) {
                net_res_stream.close();
                continue;
            }
            while (!net_res_stream.eof()) {
                std::string data;
                net_res_stream >> data;
                net_res += data;
            }
            net_res_stream.close();

            /* This device is not bound to the IB one */
            if (ib_res.compare(net_res) != 0) continue;

            /* We found the net device corresponding to the IB one, return */
            net_iface = curr_iface;
            break;
        }
    }

    return net_iface;
}

uint32_t get_dev_ip_addr(ibv_device *device) {
    std::string net_dev = get_dev_net_iface(device);
    if (net_dev.empty()) return 0;

    int fd = socket(AF_INET, SOCK_DGRAM, 0);

    auto *ifr = new ifreq;
    ifr->ifr_addr.sa_family = AF_INET;
    strncpy(ifr->ifr_name, net_dev.c_str(), IFNAMSIZ - 1);
    ioctl(fd, SIOCGIFADDR, ifr);

    close(fd);

    return ((sockaddr_in *)&ifr->ifr_addr)->sin_addr.s_addr;
}
ibv_qp * create_qp(uint32_t max_send_wr, uint32_t max_recv_wr,
                   uint32_t max_send_sge, uint32_t max_recv_sge,
                   ibv_cq * cq,
                   ibv_qp_type type,
                   ibv_pd * pd) {
    return create_qp(max_send_wr, max_recv_wr,max_send_sge, max_recv_sge, cq,cq, type, pd);
}
ibv_qp * create_qp(uint32_t max_send_wr, uint32_t max_recv_wr,
                                  uint32_t max_send_sge, uint32_t max_recv_sge,
								  ibv_cq * send_cq,
                                  ibv_cq * recv_cq,
								  ibv_qp_type type,
								  ibv_pd * pd)
{
    LOG(INFO) << "Creating QP with max_send_wr=" << max_send_wr
              << ", max_recv_wr=" << max_recv_wr << ", "
              << "max_send_sge=" << max_send_sge
              << ", max_recv_sge=" << max_recv_sge;

    auto *qp_init_attrs = new ibv_qp_init_attr;
    memset(qp_init_attrs, 0, sizeof(ibv_qp_init_attr));

    qp_init_attrs->send_cq = send_cq;
    qp_init_attrs->recv_cq = recv_cq;
    qp_init_attrs->qp_type = type;
    qp_init_attrs->cap.max_send_wr = max_send_wr;
    qp_init_attrs->cap.max_recv_wr = max_recv_wr;
    qp_init_attrs->cap.max_send_sge = max_send_sge;
    qp_init_attrs->cap.max_recv_sge = max_recv_sge;
    qp_init_attrs->sq_sig_all=0;

    ibv_qp *qp = ibv_create_qp(pd, qp_init_attrs);
    delete qp_init_attrs;

    if(!qp) {
        LOG(ERROR) << "Failed to create QP: " << strerror(errno);
        return nullptr;
    }

    LOG(INFO) << "QP created successfully with QP Num = " << qp->qp_num;

    return qp;
}

void qp_to_init(ibv_qp *qp, uint8_t port_num,
                                uint access_flags)
{
    LOG(INFO) << "Moving QP=" << qp->qp_num << " into INIT state...";

    auto *init_attrs = new ibv_qp_attr;
    memset(init_attrs, 0, sizeof(ibv_qp_attr));
    init_attrs->qp_state = ibv_qp_state::IBV_QPS_INIT;
    init_attrs->port_num = port_num;
    init_attrs->pkey_index = 0;
    init_attrs->qp_access_flags = access_flags;
    

    int flags = 0;
        flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT
                       | IBV_QP_ACCESS_FLAGS;


    if(ibv_modify_qp(qp, init_attrs, flags))
        LOG(ERROR) << "Failed to modify QP=" << qp->qp_num
                   << " to INIT: " << strerror(errno);

    delete init_attrs;
}

void qp_to_rtr(ibv_qp *qp, ibv_mtu mtu, uint32_t dest_qp,
                               ibv_ah_attr attrs)
{
    LOG(INFO) << "Moving QP=" << qp->qp_num << " into RTR state...";

    auto *rtr_attrs = new ibv_qp_attr;
    memset(rtr_attrs, 0, sizeof(ibv_qp_attr));
    rtr_attrs->qp_state = ibv_qp_state::IBV_QPS_RTR;
    rtr_attrs->path_mtu = mtu;
    rtr_attrs->dest_qp_num = dest_qp;
    rtr_attrs->rq_psn = 0;
    rtr_attrs->max_dest_rd_atomic = 16;
    rtr_attrs->min_rnr_timer = 0;
    rtr_attrs->ah_attr = attrs;

    int flags = IBV_QP_STATE
					 | IBV_QP_AV
					 | IBV_QP_PATH_MTU
					 | IBV_QP_DEST_QPN
                     | IBV_QP_RQ_PSN;
    if (qp->qp_type == IBV_QPT_RC)
        flags |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;



    if(ibv_modify_qp(qp, rtr_attrs, flags))
        LOG(ERROR) << "Failed to modify QP=" << qp->qp_num
                   << " to RTR: " << strerror(errno);

    delete rtr_attrs;
}

void qp_to_rts(ibv_qp *qp)
{
    LOG(INFO) << "Moving QP=" << qp->qp_num << " into RTS state...";

    auto *rts_attrs = new ibv_qp_attr;
    memset(rts_attrs, 0, sizeof(ibv_qp_attr));
    rts_attrs->qp_state = ibv_qp_state::IBV_QPS_RTS;

    rts_attrs->timeout = 0;
    rts_attrs->retry_cnt = 7;
    rts_attrs->rnr_retry = 7;
    rts_attrs->sq_psn = 0;
    rts_attrs->max_rd_atomic = 16;

    int flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
    if (qp->qp_type == IBV_QPT_RC)
					 flags |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT
                       | IBV_QP_RNR_RETRY
                       | IBV_QP_MAX_QP_RD_ATOMIC;

    if(ibv_modify_qp(qp, rts_attrs, flags))
        LOG(ERROR) << "Failed to modify QP=" << qp->qp_num
                   << " to RTS: " << strerror(errno);

    delete rts_attrs;
}


int finalize_rdma_connection(ibv_qp * qp, ibv_qp_attr * qp_attrs, int dest_qp)
{
    qp_to_rtr(qp, IBV_MTU_4096, dest_qp, qp_attrs->ah_attr);
    qp_to_rts(qp);
    return 0;
}

ibv_cq * create_cq(int cqe, ibv_context * ctx, ibv_comp_channel * completion_channel)
{
    LOG(INFO) << "Creating Completion Queue...";
    ibv_cq *cq = ibv_create_cq(ctx, cqe, nullptr, completion_channel, 0);

    if(!cq) {
        ibv_destroy_comp_channel(completion_channel);

        LOG(ERROR) << "Failed to create CQ: " << strerror(errno);
        return nullptr;
    }

    return cq;
}

int rdma_write_standard(ibv_mr * mr, ibv_qp * qp) {

	 ibv_send_wr * bad_wr;
	 ibv_send_wr wr;
	 ibv_sge sge;
	 memset(&wr, 0, sizeof(wr));  
	 memset(&sge, 0, sizeof(sge));  
	 sge.addr = (uint64_t) mr->addr;
	 sge.length = mr->length;
	 sge.lkey = mr->lkey;

	 wr.num_sge=1;
	 wr.sg_list = &sge;
	 wr.opcode = IBV_WR_RDMA_WRITE;
	 wr.wr.rdma.rkey = 0x1234;
	 wr.wr.rdma.remote_addr = (uint64_t) mr->addr;
	 wr.wr_id = 0x1717;
	 wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_SOLICITED | IBV_SEND_FENCE;


	return ibv_post_send(qp, &wr, &bad_wr);
}
int rdma_send_standard(ibv_mr * mr, ibv_qp * qp) {

	 ibv_send_wr * bad_wr;
	 ibv_send_wr wr;
	 ibv_sge sge;
	 memset(&wr, 0, sizeof(wr));  
	 memset(&sge, 0, sizeof(sge));  
	 sge.addr = (uint64_t) mr->addr;
	 sge.length = mr->length;
	 sge.lkey = mr->lkey;

	 wr.num_sge=0;
	 wr.opcode = IBV_WR_SEND_WITH_IMM;
	 wr.wr_id = 0x1717;
	 //wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_SOLICITED | IBV_SEND_FENCE;
	 wr.send_flags = IBV_SEND_SOLICITED | IBV_SEND_FENCE;
     wr.imm_data = 0x12345678;


	return ibv_post_send(qp, &wr, &bad_wr);
}


int rdma_write_imm_standard(ibv_qp *qp, void * buffer, size_t size,
                                                uint32_t buffer_lkey,
        uint32_t rkey, void * raddr,
                                                int imm, bool signaled){

	 ibv_send_wr * bad_wr;
	 ibv_send_wr wr;
     ibv_sge sglist;

	 memset(&wr, 0, sizeof(ibv_send_wr));

     if (buffer != nullptr)
     {
	     memset(&sglist, 0, sizeof(ibv_sge));
         wr.num_sge=1;
         wr.sg_list = &sglist;

         sglist.addr = (uint64_t) buffer;
         sglist.length = size;
         sglist.lkey = buffer_lkey;
     }
     wr.wr.rdma.remote_addr = (uint64_t) raddr;
     wr.wr.rdma.rkey =  rkey;

	 wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	 wr.wr_id = imm;

	 wr.send_flags = signaled ? (IBV_SEND_SIGNALED | IBV_SEND_SOLICITED) : 0; // IBV_SEND_SOLICITED | IBV_SEND_FENCE;
     wr.imm_data = imm;

	// printf("Sending a wr with sge at %p with %i wr\n",
            // wr.sg_list, wr.num_sge);
    int ret = ibv_post_send(qp, &wr, &bad_wr);
     //printf("ibv_post_send returned %i\n", ret);
    return ret;
}




int post_recv(ibv_qp * qp)
{

    struct ibv_recv_wr wr;
    struct ibv_recv_wr *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id      = 0;
    wr.sg_list    = nullptr;
    wr.num_sge    = 0;

    int ret = ibv_post_recv(qp, &wr, &bad_wr);
    return ret;
}
