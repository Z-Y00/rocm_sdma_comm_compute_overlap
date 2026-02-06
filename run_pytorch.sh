#!/bin/bash

export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1235

# rccl install: 
# ./install.sh -f
# export NCCL_P2P_USE_CUDA_MEMCPY=1
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true 
export NCCL_DEBUG=INFO
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

torchrun "${DISTRIBUTED_ARGS[@]}" ./corun_test_rccl.py $1 $2 $3 $4 $5 $6
