git submodule init
git submodule sync
git submodule update --recursive
cd gflags
cmake . -DBUILD_SHARED_LIBS=ON
make -j
cd ..

cd glog
cmake . -DCMAKE_PREFIX_PATH=../gflags
make -j
cd ..

git clone https://github.com/grpc/grpc.git --recursive --branch v1.48.0
cd grpc
git checkout v1.48.0 
# git submodule init
# git submodule update --recursive
# mkdir -p build && cd build
# cmake -DgRPC_INSTALL=ON \
#       -DgRPC_BUILD_TESTS=OFF \
#       -DCMAKE_INSTALL_PREFIX=$(pwd)/../install \
#       ..
# make -j -C build
# make -j -C build install

mkdir build
cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$(pwd)/install -B build
make -j -C build
make -j -C build install

