# Init the testing env
Targeting PR:

https://github.com/ROCm/rocm-systems/pull/2663/changes
## Step 1 start the docker
```
docker run -it --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME:$HOME \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size 64G \
    --name training_env \
    rocm/primus:v26.1
```

## Step 2 build the lib
Ref: https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html
```
apt-get install libsimde-dev xxd
pip3 install CppHeaderParser
apt-get install rocm-llvm-dev 

git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems

git checkout release/rocm-rel-7.1.1.1

git cherry-pick a8c1273cfaee04e8e3252d75d557451431ef94b0
cd -

# build the hip and clr
export CLR_DIR="$(readlink -f rocm-systems/projects/clr)"
export HIP_DIR="$(readlink -f rocm-systems/projects/hip)"
cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF ..
make -j$(nproc)
sudo make install
```

## Step3 Testing
```
# clone
git clone https://github.com/Z-Y00/rocm_sdma_comm_compute_overlap.git
cd rocm_sdma_comm_compute_overlap

# build
hipcc -std=c++14 -O3 -I/opt/rocm/hip/include -o sdma_memcpy_test sdma_memcpy_test.cpp -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi

# test 
HSA_ENABLE_SDMA=1 mpirun -np 8 --allow-run-as-root sdma_memcpy_test
HSA_ENABLE_SDMA=1 AMD_LOG_LEVEL=4 mpirun -np 8 --allow-run-as-root sdma_memcpy_test > debug.txt 2>&1
cat ./debug.txt  | grep forceSD
```


Ddebug note
```
# MPI compile debug
mpicc -showme:compile
-I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi
```

## Step4 Profiling
We need to make sure that this program is really using sdma, not regular memcpy.
```
rocprofv3 --kernel-trace --output-format csv -- mpirun -np 8 --allow-run-as-root sdma_memcpy_test
```
We should see 4 __amd_rocclr_copyBuffer, which are for the h2d,d2h memcopy
