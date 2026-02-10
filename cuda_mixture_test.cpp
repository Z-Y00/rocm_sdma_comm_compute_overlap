#include <cuda_runtime.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CUDA_CHECK(cmd)                                                        \
    {                                                                          \
        cudaError_t error = cmd;                                               \
        if (error != cudaSuccess) {                                            \
            fprintf(stderr, "Error: '%s'(%d) at %s:%d\n",                      \
                    cudaGetErrorString(error), error, __FILE__, __LINE__);     \
            MPI_Abort(MPI_COMM_WORLD, error);                                  \
        }                                                                      \
    }

#define MPI_CHECK(cmd)                                                         \
    {                                                                          \
        int error = cmd;                                                       \
        if (error != MPI_SUCCESS) {                                            \
            fprintf(stderr, "MPI Error: %d at %s:%d\n", error, __FILE__,       \
                    __LINE__);                                                 \
            MPI_Abort(MPI_COMM_WORLD, error);                                  \
        }                                                                      \
    }

constexpr int NELT = 1024 * 1024;
constexpr size_t BUF_SIZE = NELT * sizeof(float);

static void run_test(bool use_pinned_host) {
    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    const int peer_rank = 1 - world_rank;
    const int my_device = world_rank;
    const int peer_device = peer_rank;

    // Host buffers: pinned (test 1) or unpinned (test 2)
    float* h_src = nullptr;
    float* h_dst = nullptr;
    if (use_pinned_host) {
        CUDA_CHECK(cudaMallocHost(&h_src, BUF_SIZE));
        CUDA_CHECK(cudaMallocHost(&h_dst, BUF_SIZE));
    } else {
        h_src = (float*)malloc(BUF_SIZE);
        h_dst = (float*)malloc(BUF_SIZE);
        if (!h_src || !h_dst) {
            fprintf(stderr, "Rank %d: malloc failed\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    for (int i = 0; i < NELT; i++)
        h_src[i] = world_rank * 1000.0f + static_cast<float>(i);

    float* d_buf1 = nullptr;
    float* d_buf2 = nullptr;
    float* d_p2p = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf1, BUF_SIZE));
    CUDA_CHECK(cudaMalloc(&d_buf2, BUF_SIZE));
    CUDA_CHECK(cudaMalloc(&d_p2p, BUF_SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Exchange IPC handles for d_buf2 (source for P2P send) and d_p2p (dest for P2P recv)
    cudaIpcMemHandle_t handle_buf2, handle_p2p;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle_buf2, d_buf2));
    CUDA_CHECK(cudaIpcGetMemHandle(&handle_p2p, d_p2p));

    cudaIpcMemHandle_t peer_handle_buf2, peer_handle_p2p;
    MPI_CHECK(MPI_Sendrecv(&handle_buf2, sizeof(cudaIpcMemHandle_t), MPI_BYTE, peer_rank, 0,
                           &peer_handle_buf2, sizeof(cudaIpcMemHandle_t), MPI_BYTE, peer_rank, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Sendrecv(&handle_p2p, sizeof(cudaIpcMemHandle_t), MPI_BYTE, peer_rank, 1,
                           &peer_handle_p2p, sizeof(cudaIpcMemHandle_t), MPI_BYTE, peer_rank, 1,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    void* peer_d_buf2 = nullptr;
    void* peer_d_p2p = nullptr;
    CUDA_CHECK(cudaIpcOpenMemHandle(&peer_d_buf2, peer_handle_buf2, cudaIpcMemLazyEnablePeerAccess));
    CUDA_CHECK(cudaIpcOpenMemHandle(&peer_d_p2p, peer_handle_p2p, cudaIpcMemLazyEnablePeerAccess));

    // Batched memcpy: 1) HtoD  2) DtoD  3) P2P  4) D2H
    constexpr size_t num_copies = 4;
    const void* src_ptrs[num_copies] = { h_src, d_buf1, d_buf2, d_p2p };
    void* dst_ptrs[num_copies] = { d_buf1, d_buf2, peer_d_p2p, h_dst };
    size_t sizes[num_copies] = { BUF_SIZE, BUF_SIZE, BUF_SIZE, BUF_SIZE };

    cudaMemcpyAttributes attrs[1] = {};
    attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs[0].flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attrIdxs[] = { 0, num_copies };

    CUDA_CHECK(cudaMemcpyBatchAsync(dst_ptrs, src_ptrs, sizes, num_copies, attrs, attrIdxs, 1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup
    CUDA_CHECK(cudaIpcCloseMemHandle(peer_d_buf2));
    CUDA_CHECK(cudaIpcCloseMemHandle(peer_d_p2p));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaFree(d_buf2));
    CUDA_CHECK(cudaFree(d_p2p));
    if (use_pinned_host) {
        CUDA_CHECK(cudaFreeHost(h_src));
        CUDA_CHECK(cudaFreeHost(h_dst));
    } else {
        free(h_src);
        free(h_dst);
    }
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    if (world_size != 2) {
        if (world_rank == 0)
            fprintf(stderr, "This program requires exactly 2 MPI processes\n");
        MPI_Finalize();
        return 1;
    }

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        if (world_rank == 0)
            fprintf(stderr, "This program requires at least 2 GPUs, found %d\n", device_count);
        MPI_Finalize();
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(world_rank));

    // Enable P2P access for cross-device copy
    int can_access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, world_rank, 1 - world_rank));
    if (can_access)
        {CUDA_CHECK(cudaDeviceEnablePeerAccess(1 - world_rank, 0));}
    else if (world_rank == 0)
        fprintf(stderr, "Warning: P2P access not available between devices\n");

    if (world_rank == 0)
        printf("CUDA mixture test: 2 ranks, HtoD -> DtoD -> P2P -> D2H\n");

    MPI_Barrier(MPI_COMM_WORLD);

    // Test 1: pinned host buffers
    if (world_rank == 0)
        printf("\n--- Test 1: Pinned host buffers (H2D and D2H) ---\n");
    MPI_Barrier(MPI_COMM_WORLD);
    run_test(true);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test 2: unpinned host buffers
    if (world_rank == 0)
        printf("\n--- Test 2: Unpinned host buffers (H2D and D2H) ---\n");
    MPI_Barrier(MPI_COMM_WORLD);
    run_test(false);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0)
        printf("\nAll tests completed.\n");

    MPI_Finalize();
    return 0;
}
