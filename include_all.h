#ifndef APPS_INCLUDE_ALL_H
#define APPS_INCLUDE_ALL_H

// As a bad practice, we import all libraries here.
// So we have a single file for all apps.
// This of course means that also clients would have to be aware of CUDA headers!


#include <arpa/inet.h>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <net/if.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <signal.h>
#include <boost/program_options.hpp>
#include <driver_types.h>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <unistd.h>
#include <vector>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>




#include <cuda_utils.cuh>
extern "C" {
#include <rdma_shim.h>
}

// using namespace std;

#endif
