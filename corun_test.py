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

    # Set up symmetric memory with NCCL backend
    symm_mem.set_backend("NCCL")
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    args._symm_group_name = group_name

    if args.rank == 0:
        print(
            f"[Rank {args.rank}][LocalRank {args.local_rank}] Initialized with world size {args.world_size} (CE collectives + symmetric memory)"
        )


def bandwidth_test(rank, size, world_size, group_name, comm):
    """
    Run bandwidth test using symmetric memory tensors and Copy Engine collectives.
    Communication runs on DMA engines instead of SMs for better compute/comm overlap.
    """
    device = torch.device("cuda", rank)
    s = torch.cuda.Stream(device=device)
    print("starting test")
    num_elems = int(size)

    # Allocate tensors using symmetric memory
    # inp_symm = symm_mem.empty(num_elems, device=device)
    # out_symm = symm_mem.empty(num_elems * world_size, device=device)


    # Cast to desired dtype and fill input (symm_mem.empty returns a tensor we can use)
    # inp_symm = inp_symm.to(torch.bfloat16)
    # out_symm = out_symm.to(torch.bfloat16)
    # inp_symm.fill_(rank)

    # # Register tensors for symmetric memory operations
    # symm_mem.rendezvous(inp_symm, group=group_name)
    # symm_mem.rendezvous(out_symm, group=group_name)

    # Normal tensor allocation
    input_tensor = torch.full([num_elems], rank, dtype=torch.bfloat16, device=device)
    output_tensor = torch.zeros([num_elems * world_size], dtype=torch.bfloat16, device=device)
 
    # dist.barrier(async_op=False)
    torch.cuda.synchronize(device=rank)
    start_time = time.time()

    # Compute on a separate stream (matmul)
    # m, n, k = 10240, 24576, 8192
    # A = torch.randn(k, m, dtype=torch.bfloat16, device=device)
    # B = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    # A = A.t()
    # m, n, k = 16384, 53248, 16384
    torch.cuda.synchronize(device=rank)

    # with torch.cuda.stream(s):
    #     for i in range(15):
    #         torch.matmul(A, B)

    time.sleep(1)

    # Perform collective using copy engines (async_op=True required for CE collectives)
    if comm == "all_gather":
        print("Doing all_gather")
        for i in range(1):
            # work = dist.all_gather_into_tensor(out_symm, inp_symm, async_op=True)
            # work.wait()
            work = dist.all_gather_into_tensor(output_tensor,input_tensor, async_op=True)
            work.wait()
    else:
        print("Doing reduce_scatter")
        for i in range(1):
            # work = dist.reduce_scatter_tensor(
                # inp_symm, out_symm, torch.distributed.ReduceOp.AVG, async_op=True
            # )
            work.wait()

    torch.cuda.synchronize(device=rank)
    end_time = time.time()
    elapsed_time = end_time - start_time


def run(rank, size, config, comm, group_name):
    """Distributed run: bandwidth test with symmetric memory and CE collectives."""
    device = torch.device("cuda", rank)
    # dist.barrier(async_op=False)
    torch.cuda.synchronize(device=rank)
    print("test run finish")

    bandwidth_test(rank, (1024 * 1024 * 204) // 8, size, group_name, comm)
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        # bandwidth_test(None, rank, (1024*1024*204/8), size,config,comm)
        # bandwidth_test(rank, (1024*1024*1024/2), size)
    # prof.export_chrome_trace(f"./trace.{rank}.json")
    torch.cuda.synchronize(device=rank)


if __name__ == "__main__":
    args = get_args()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    init_dist(args)

    if args.dtype == "16":
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float8_e4m3fnuz)

    run(
        args.rank,
        args.world_size,
        args.config,
        args.comm,
        getattr(args, "_symm_group_name", dist.group.WORLD.group_name),
    )

    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"{args.rank=}, success.", flush=True)
