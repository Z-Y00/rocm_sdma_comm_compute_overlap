"""
Distributed bandwidth test using Copy Engine (CE) collectives with symmetric memory.
See: https://docs.pytorch.org/docs/stable/distributed.html#copy-engine-collectives

Requirements: NCCL 2.28+, GPUs with P2P access.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed.distributed_c10d as c10d

import argparse
from datetime import timedelta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", "0")),
        help="local rank passed from distributed launcher.",
    )
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--dtype", type=str, default="16")
    parser.add_argument("--comm", type=str, default="all_gather")
    parser.add_argument("--enable-profile", action="store_true", required=False)
    parser.add_argument("--profile-step-start", type=int, default=0)
    parser.add_argument("--profile-step-end", type=int, default=8)
    parser.add_argument(
        "--distributed-timeout-minutes", type=int, default=10,
        help="Timeout minutes for torch.distributed.",
    )
    return parser.parse_args()


def init_dist(args):
    """Initialize process group with zero-CTA policy for CE collectives and symmetric memory."""
    torch.cuda.device_count()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    pool = torch.cuda.MemPool(allocator=torch.cuda.get_memory_allocator(device), device=device)

    # Initialize process group with zero-CTA policy for Copy Engine collectives
    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO

    dist.init_process_group(
        backend=args.backend,
        pg_options=opts,
        device_id=device,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    pg_nccl = dist.group.WORLD  # if backend is NCCL, this is the right object

    # Register the pool with symm=True so that registerSegment(..., symm=true) is used
    pg_nccl.register_mem_pool(pool, symm=True)
    rank = args.rank
    world_size = args.world_size
    device = torch.device("cuda", rank)
    s = torch.cuda.Stream(device=device)
    print("starting test")
    size = 1024*1024
    num_elems = int(size)

    with torch.cuda.use_mem_pool(pool):
        input_tensor = torch.full([num_elems], rank, dtype=torch.bfloat16, device=device)
        output_tensor = torch.zeros([num_elems * world_size], dtype=torch.bfloat16, device=device)
    # dist.barrier(async_op=False)
    torch.cuda.synchronize(device=rank)
    print("Doing all_gather")
    for i in range(1):
            # work = dist.all_gather_into_tensor(out_symm, inp_symm, async_op=True)
            # work.wait()
            work = dist.all_gather_into_tensor(output_tensor,input_tensor, async_op=True)
            work.wait()





if __name__ == "__main__":
    args = get_args()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    init_dist(args)

    if args.dtype == "16":
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float8_e4m3fnuz)
 
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"{args.rank=}, success.", flush=True)
