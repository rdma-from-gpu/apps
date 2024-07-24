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

using namespace std;

#endif
