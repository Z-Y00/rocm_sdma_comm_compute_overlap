#!/bin/bash

export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1235
export HSA_NO_SCRATCH_RECLAIM=1

export NCCL_LOCAL_REGISTER=2 # for buffer registering 
export NCCL_CTA_POLICY=2 #  NCCL_CTA_POLICY_ZERO
export NCCL_CUMEM_ENABLE=1 # default rccl value is 0 as disabling
export NCCL_DEBUG=TRACE #VERSION #
# export NCCL_P2P_USE_CUDA_MEMCPY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# torchrun "${DISTRIBUTED_ARGS[@]}" ./test_dma_p2p.py \
    # --enable-profile
# torchrun "${DISTRIBUTED_ARGS[@]}" ./test_dma_all_gather.py
# torchrun "${DISTRIBUTED_ARGS[@]}" ./test_dma_all_gather.py --enable-profile
# torchrun "${DISTRIBUTED_ARGS[@]}" ./test_symmetric_memory.py

#     --enable-profile

## Example commands
# bash ./run_rccl.sh # run 70B with all gather
# bash ./run_rccl.sh --config 405B # run 405B with all gather
# bash ./run_rccl.sh --config 405B --comm reduce_scatter # run 405B with reduce_scatter
# bash ./run_rccl.sh --comm reduce_scatter # run 70B with reduce_scatter

# Lorri, this will call to ncclCommRegister / Window Register
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
torchrun "${DISTRIBUTED_ARGS[@]}" ./corun_register_mempool.py $1 $2 $3 $4 $5 $6 > log.mempool 2>&1


# Lorri, this will call to ncclCommRegister / Window Register
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
torchrun "${DISTRIBUTED_ARGS[@]}" ./corun_test.py $1 $2 $3 $4 $5 $6 > log.normal.tensor 2>&1

# Lorri, this will call to p2p in rccl as expected
# TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false \
# torchrun "${DISTRIBUTED_ARGS[@]}" ./corun_test_symm.py $1 $2 $3 $4 $5 $6 > log.explicit.symm.tensor 2>&1
