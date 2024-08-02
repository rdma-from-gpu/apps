cmake_minimum_required(VERSION 3.4.1)
project(rdma_from_gpu)

## Find and setup CUDA at the very beginning
## TODO Use parameters
set(ENV{CUDA_PATH} "/usr/local/cuda")
#set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "75;80;89")
set(CUDA_ARCHITECTURES "75")
#find_package(CUDA)
#find_package(CUDAToolkit)

project(rdma_from_gpu LANGUAGES  C CXX CUDA)
#set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fdiagnostics-color=auto")
#set(CXX "g++-8")
#set(CC "gcc-8")
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "" OR (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else ()
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -g -fno-inline")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -g")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    endif ()
endif ()

option(NO_OMIT_FRAME_POINTER "Pass -no-omit-frame-pointer to compiler. You'd like to want that for profiling" OFF)
if (NO_OMIT_FRAME_POINTER)
    message("Not omitting Frame Pointer (a.k.a. using 1 more CPU register in function calls...)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
endif ()
unset(NO_OMIT_FRAME_POINTER CACHE)


option(CCACHE "Accelerate compilation with ccache" ON)
if (CCACHE)
    find_program(CCACHE_FOUND ccache)
    if (CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        message("Speeding up compilation (hopefully) via ccache")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    endif (CCACHE_FOUND)
endif(CCACHE)
unset(CCACHE CACHE)

option(CUDA_TRACING "Enable NVTX calls to trace and profile with CUDA Nsight" OFF)
if (CUDA_TRACING)
        message("Enable NVTX calls to trace and profile with CUDA Nsight")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_TRACING=1")
        #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DCUDA_TRACING=1")
        add_compile_definitions("CUDA_TRACING=1")
endif ()
unset(CUDA_TRACING CACHE)


# Look for TVM in ./tvm and ../tvm, but only if TVM_HOME is not set
if ("$ENV{TVM_HOME}" STREQUAL "")
    find_path(TVM_HOME include/tvm/runtime/device_api.h HINTS ./tvm ../tvm ../../tvm NOCACHE)
else ()
    set(TVM_HOME $ENV{TVM_HOME})
endif ()
message("TVM found at ${TVM_HOME}")
if ("$ENV{RDMA_CORE_HOME}" STREQUAL "")
    find_path(RDMA_CORE_HOME ../rdma-core/providers/mlx5/mlx5.c HINTS ./rdma-core ../rdma-core NOCACHE)
else ()
    set(RDMA_CORE_HOME $ENV{RDMA_CORE_HOME})
endif ()
message("rdma-core found at ${RDMA_CORE_HOME}")
if ("$ENV{RDMA_SHIM_HOME}" STREQUAL "")
    find_path(RDMA_SHIM_HOME ./build/librdma-shim-cuda.a HINTS ./rdma-shim ../rdma-shim NOCACHE)
else ()
    set(RDMA_SHIM_HOME $ENV{RDMA_SHIM_HOME})
endif ()
message("rdma-shim found at ${RDMA_SHIM_HOME}")
message("rdma-shim include found at ${RDMA_SHIM_INCLUDE}")


#set(CUDA_ARCHITECTURES "75;80;89" CACHE STRING "CUDA architectures to pass to NVCC")
set(CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures to pass to NVCC")

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Build type: " ${CMAKE_BUILD_TYPE})
message("CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message("Using NVCC at ${CMAKE_CUDA_COMPILER}")
message("NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message("Using host compiler at ${CMAKE_CUDA_HOST_COMPILER}")
message("CUDA architectures: ${CUDA_ARCHITECTURES}")
message("Prepending compile commands with ${RULE_LAUNCH_COMPILE}")

find_package(Threads REQUIRED)
find_package(PkgConfig)
find_package(gflags REQUIRED HINTS ./extern/gflags)
find_package(glog 0.6.0 REQUIRED HINTS ./extern/glog)
find_package( Boost 1.30 COMPONENTS program_options REQUIRED )
find_package(absl CONFIG REQUIRED HINTS extern/grpc/install/lib/cmake/absl/ )
find_package(Protobuf CONFIG REQUIRED HINTS extern/grpc/install/lib/cmake/protobuf)
find_package(gRPC CONFIG REQUIRED HINTS extern/grpc/install)

set(RDMA_CORE_LIB ${RDMA_CORE_HOME}/build/lib)
set(RDMA_CORE_INCLUDE ${RDMA_CORE_HOME}/build/include)
set(RDMA_CORE_INCLUDE_MLX5 ${RDMA_CORE_HOME}/providers/mlx5/)
set(RDMA_CORE_INCLUDE_SRC ${RDMA_CORE_HOME}/)
set(RDMA_SHIM_INCLUDE ${RDMA_SHIM_HOME})
set(RDMA_SHIM_LIB ${RDMA_SHIM_HOME}/build/)

link_directories(
    ${RDMA_CORE_LIB}
    ${RDMA_SHIM_LIB}
)


# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)
# We have sub-projects, but we want CMAKE to put all binaries together
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


add_library(rdma-apps STATIC network.cpp generator-verbs.cu)
set_target_properties(rdma-apps PROPERTIES
    LANGUAGE CPP)

target_include_directories(rdma-apps PUBLIC
    ${RDMA_SHIM_INCLUDE}
    ${RDMA_CORE_INCLUDE}
    ${RDMA_CORE_INCLUDE_SRC} 
    ${RDMA_CORE_INCLUDE_MLX5})

target_link_libraries(rdma-apps
    glog::glog
    rdma-shim
    mlx5
    ibverbs)



add_executable(generator generator.cu ${COMMON_SRCS})
set_target_properties(generator PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
target_link_libraries(generator 
    # rdma-shim_cuda
    rdma-apps
    rdma-shim
    rdma-shim-cuda
    glog::glog
    mlx5
    cuda
    nvToolsExt
    ${Boost_LIBRARIES}
   )
target_include_directories(generator PUBLIC
    glog::glog
    ${RDMA_CORE_INCLUDE}
    ${RDMA_SHIM_INCLUDE}
    ${RDMA_CORE_INCLUDE_MLX5})

