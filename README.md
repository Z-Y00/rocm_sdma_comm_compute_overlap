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
Patch: https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html
```
apt-get install libsimde-dev xxd rocm-llvm-dev vim -y
pip3 install CppHeaderParser

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
make install
```

## Step3.1 Testing with simple program (skipable after init testing)
```
# clone
cd /workspace
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

## Step3.2 Profiling simple program
We need to make sure that this program is really using sdma, not regular memcpy.
```
rocprofv3 --kernel-trace --output-format csv -- mpirun -np 8 --allow-run-as-root sdma_memcpy_test
```
We should see 4 __amd_rocclr_copyBuffer, which are for the h2d,d2h memcopy

## Step4 RCCL build and test
```
cd /workspace
git clone https://github.com/Z-Y00/rccl-sdma.git
cd rccl-sdma
./install.sh -if # --generate-sym-kernels # we need to explicit setting this for some commits

```
### Test and Profile rccl
```
# Note: Please check the RCCL version printed, in case the older one didn't got replaced 
export NCCL_DEBUG=VERSION
# run any rccl-test commmand
```
### replace python
[This?](https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html#using-rccl-and-cpx-in-pytorch)
Librccl path :
cp /opt/rocm/lib/librccl.so /opt/venv/lib/python3.10/site-packages/torch/lib/librccl.so
nm /opt/rocm/lib/librccl.so | grep ncclSymkGetKernelPtr
### pytorch patch
https://docs.pytorch.org/docs/stable/distributed.html#copy-engine-collectives

We also need to recompile pytorch, as it will detect rccl CE support before kicking start.
https://github.com/pytorch/pytorch/blob/6e866c4a69cb9ed0fc58e0fb20628a6e4a65e39b/torch/csrc/distributed/c10d/NCCLUtils.hpp#L65

related code
https://github.com/pytorch/pytorch/blob/af3ee4dc21caa68d860acf0f8f66064469066a8a/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L242
https://github.com/search?q=repo%3Apytorch%2Fpytorch+registerSegment+&type=code
https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAAllocatorConfig.h#L34
https://github.com/pytorch/pytorch/blob/af3ee4dc21caa68d860acf0f8f66064469066a8a/torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu#L87
https://github.com/pytorch/pytorch/blob/af3ee4dc21caa68d860acf0f8f66064469066a8a/torch/csrc/distributed/c10d/NCCLUtils.cpp#L501
```
cd /workspace
python -c "import torch; print(torch.version.git_version)"
# 7bb0466bb4d732f0aa3273c0122f3213003b182b
python3 -m pip install --upgrade pip
pip install ninja  setuptools wheel uv tabulate ipython pytest fire pydantic pybind11
pip install cmake==4.2.1
pip install setuptools==82.0.0
git clone https://github.com/pytorch/pytorch.git \
    && cd pytorch \
    && git checkout 7bb0466bb4d732f0aa3273c0122f3213003b182b \
    && git submodule update --recursive --init \
    && ./tools/amd_build/build_amd.py \
    && BUILD_TEST=0 python3 setup.py install \
    && cd ..

python -c "import torch; print(torch.distributed.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO)"
# 2

docker commit -m "fix pytorch upgrade" 0f1b94036eff sdma

# test pytorch rccl
bash ./run_pytorch.sh
```


# CUDA test
```
docker run --rm --gpus all  -it   -v $HOME:$HOME    -v $HOME/.ssh:/root/.ssh  nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 bash
apt update
apt-get install gdb openmpi-bin libopenmpi-dev vim -y
 mpicc -showme:compile
-I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi

nvcc -std=c++14 -O0 -g -o sdma_memcpy_test test.cpp -lmpi -lmpi_cxx -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi

gdb --args mpirun -np 8 --allow-run-as-root sdma_memcpy_test

nsys profile mpirun -np 8 --allow-run-as-root sdma_memcpy_test
nsys profile mpirun -np 2 --allow-run-as-root sdma_memcpy_test
```