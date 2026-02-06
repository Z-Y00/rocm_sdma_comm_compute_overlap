import os
import time
import torch
import datetime
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.profiler import profile, record_function, ProfilerActivity
# from ipc_dma.all_gather import AllGatherEngine
#dma_engine = AllGatherEngine("test_dma_all_gather")


import argparse
from datetime import timedelta
import torch


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
        "--distributed-timeout-minutes", type=int, default=10, help="Timeout minutes for torch.distributed."
    )
    return parser.parse_args()


def init_dist(args):
    torch.cuda.device_count()
    torch.cuda.set_device(args.local_rank)
    if args.rank == 0:
        print(
            f"[Rank {args.rank}][LocalRank {args.local_rank}] Initialized with world size {args.world_size}"
        )

    dist.init_process_group(
        backend=args.backend,
        device_id=torch.device(f"cuda:{args.local_rank}"),
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )



tag = "test_dma_all_gather"
def run_gemm(device):
        m=28672  
        n=16384  
        k=4096
        A = torch.randn(k,m).to(device)
        B = torch.randn(k,n).to(device)
        # C = torch.randn(dimC).type(dtype)
        at = True
        if at: A = A.t()
        # if bt: B = B.t()
        torch.matmul(A, B)

def mm_TFLOPS_(A,B,float_point, rank):
    iter_num = 20
    warm_up = 10
    for i in range(warm_up):
        torch.matmul(A, B)
    torch.cuda.synchronize(device=rank)
    start_time = time.time()
    for i in range(iter_num):
        torch.matmul(A, B)
    torch.cuda.synchronize(device=rank)
    end_time = time.time()
    print("stand alone TFLOPS:",float_point*iter_num/1024/1024/1024/1024/(end_time-start_time))

def bandwidth_test(dma_engine, rank, size, world_size,config,comm):
    device=torch.device(f"cuda:{rank}")
    s = torch.cuda.Stream(device=device)
    print("starting test")
    num_elems = int(size)
    input_tensor = torch.full([num_elems], rank, dtype=torch.bfloat16, device=f"cuda")
    output_tensor = torch.zeros([num_elems * world_size], dtype=torch.bfloat16, device=f"cuda")
 
    # Create a list to hold the gathered tensors.
    # tensor = torch.full([num_elems], rank).to(device)
    # size_in_gb = tensor.element_size() * tensor.numel()/1024/1024/1024
    # gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    # output_tensor = torch.zeros([num_elems*world_size]).to(device)

    # Perform the all_gather operation.
    dist.barrier(async_op=False) # wait till all the process is ready
    torch.cuda.synchronize(device=rank)
    start_time = time.time()
    m=10240
    n=24576
    k=8192 # from https://raw.githubusercontent.com/AMD-AIG-AIMA/MAD-private/83d2e61524aea7fa2e8c2ab67c66d95a2e0651a6/gemm_result/Llama-3.1-70B.BF16.default.351.log?token=GHSAT0AAAAAADAEWUEK4A6KPIRI2MFARKWI2CKB7DQ
    A = torch.randn(k,m).to(device)
    B = torch.randn(k,n).to(device)
    float_point = 2 * m * k * n
    # print("mm Gfloat:",float_point/1024/1024/1024)
    A = A.t()
    # print("normal gemm perf test")
    iter_num = 20
    warm_up = 10
    # mm_TFLOPS_(A,B,float_point, rank)
    # using the mnk posted by Joyce in the meeting chat
    m = 16384
    n = 53248
    k = 16384
    a = torch.randn(k,m).to(device) # T
    b = torch.randn(k,n).to(device) # N
    def test_405_gemm():
        torch.matmul(a.t(), b)
    # print("Setting config as:")
    # print(config)

    # dma_engine first run will create stream and sync device, warmup here
    #if True:
    #    for i in range(10):    
    #        torch.distributed.all_gather_into_tensor(output_tensor, input_tensor)
            # dma_engine.all_gather_into_tensor(output_tensor, input_tensor, None, tag)
    torch.cuda.synchronize(device=rank)


    with torch.cuda.stream(s):
        if config == "405B":
            print("testing with 405B gemm")
            for i in range(200):
                test_405_gemm()
        else:
            for i in range(1500):
                torch.matmul(A, B)
        #s_event = s.record_event()
    if True:
        # dist.all_gather_into_tensor(output_tensor, input_tensor)
        # if config == "405B":
            # time.sleep(3) 
        # else:
        time.sleep(1) 
        if comm == "all_gather":
            print("Doing all gather")
            for i in range(10):
                dist.all_gather_into_tensor(output_tensor, input_tensor)
        else:
            print("Doing reduce scatter")
            for i in range(10):
                dist.reduce_scatter_tensor(input_tensor, output_tensor,torch.distributed.ReduceOp.AVG)
    #if True:
    #    for i in range(10):    
    #        torch.distributed.all_gather_into_tensor(output_tensor, input_tensor)
            # dma_engine.all_gather_into_tensor(output_tensor, input_tensor, None, tag)
    # torch.cuda.current_stream().wait_event(s_event)
    torch.cuda.synchronize(device=rank)
    # dma_engine.stream_wait()
    # for i in range(iter_num):
    #     torch.matmul(A, B)
    torch.cuda.synchronize(device=rank)
    #dma_engine.all_gather_into_tensor(output_tensor, input_tensor, None, tag)
    #dma_engine.all_gather_into_tensor(output_tensor, tensor, None, tag)
    #run_gemm(device)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    #dist.reduce_scatter(tensor, gather_list,torch.distributed.ReduceOp.AVG)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    #dist.reduce_scatter(tensor, gather_list,torch.distributed.ReduceOp.AVG)
    #dist.all_gather(gather_list, tensor)
    #dist.all_gather(gather_list, tensor)
    torch.cuda.synchronize(device=rank)
    # print("post rccl test gemm test")
    # mm_TFLOPS_(A,B,float_point, rank)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.3f} \
        # seconds tensor size  (GB):{size_in_gb:.1f} \
        # Bandwidth(GB/s):{world_size*size_in_gb/elapsed_time:.1f}")
    # print("tensors:",gather_list)

def run(rank, size,config,comm):
    """ Distributed function to be implemented later. """
    """Demonstrates the use of all_gather."""
    device=torch.device(f"cuda:{rank}")
    overhead_test = torch.full([1, 1024], rank).to(device)
    size_in_gb = overhead_test.element_size() * overhead_test.numel()/1024/1024/1024
    #print("overhead_test size (GB):",size_in_gb)
    gather_list = [torch.zeros_like(overhead_test) for _ in range(size)]
    # Perform 1st all_gather operation.
    dist.barrier(async_op=False) # wait till all the process is up
    torch.cuda.synchronize(device=rank)
    start_time = time.time()
    print("test run finish")
    
    # warmup
    bandwidth_test(None, rank, (1024*1024*204/8), size,config,comm)
    

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        bandwidth_test(None, rank, (1024*1024*204/8), size,config,comm)
        # bandwidth_test(rank, (1024*1024*1024/2), size)
    prof.export_chrome_trace(f"./trace.{rank}.json")
    
    torch.cuda.synchronize(device=rank)
    # del dma_engine


def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, timeout=datetime.timedelta(seconds=5), world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 8

    args = get_args()

    # args set by torchrun
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    init_dist(args)
    if args.dtype=="16":
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float8_e4m3fnuz)
    run(args.rank, size,args.config,args.comm)

    # clean up torch pg resources on exit
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"{args.rank=}, success.", flush=True)
        
