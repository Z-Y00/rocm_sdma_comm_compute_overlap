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


int main(int argc, char** argv) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    if (world_size != 8) {
        if (world_rank == 0) {
            fprintf(stderr, "This program requires exactly 8 MPI processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Set device for this process
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count < 8) {
        if (world_rank == 0) {
            fprintf(stderr, "This program requires at least 8 GPUs, found %d\n",
                    device_count);
        }
        MPI_Finalize();
        return 1;
    }

    // Each process uses its rank as device ID
    CUDA_CHECK(cudaSetDevice(world_rank));

    if (world_rank == 0) {
        printf("Running CUDA IPC AllGather with 8 processes on 8 GPUs\n");
    }

    // Define 2D array dimensions for each rank
    const int rows = 1024;
    const int cols = 1024;
    const size_t local_size = rows * cols * sizeof(float);
    const size_t total_size = local_size * world_size;

    // Allocate local device memory for this rank's data
    float* d_local_data;
    CUDA_CHECK(cudaMalloc(&d_local_data, local_size));

    // Initialize local data on CPU
    std::vector<float> h_local_data(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        // Each rank initializes its data with rank-specific values
        h_local_data[i] = world_rank * 1000.0f + static_cast<float>(i);
    }

    // Copy initialized data to GPU
    CUDA_CHECK(cudaMemcpy(d_local_data, h_local_data.data(), local_size,
                          cudaMemcpyHostToDevice));

    // Create IPC handle for local data
    cudaIpcMemHandle_t local_ipc_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_ipc_handle, d_local_data));

    // Gather all IPC handles to all processes
    std::vector<cudaIpcMemHandle_t> all_ipc_handles(world_size);
    MPI_CHECK(MPI_Allgather(&local_ipc_handle, sizeof(cudaIpcMemHandle_t),
                            MPI_BYTE, all_ipc_handles.data(),
                            sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                            MPI_COMM_WORLD));

    // Allocate memory for gathered data (all ranks' data)
    float* d_gathered_data;
    CUDA_CHECK(cudaMalloc(&d_gathered_data, total_size));

    // Open IPC handles to access remote memory
    std::vector<void*> remote_ptrs(world_size);
    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) {
            // Use local pointer for own data
            remote_ptrs[i] = d_local_data;
        } else {
            // Open IPC handle for remote rank's memory
            CUDA_CHECK(cudaIpcOpenMemHandle(&remote_ptrs[i],
                                           all_ipc_handles[i],
                                           cudaIpcMemLazyEnablePeerAccess));
        }
    }

    if (world_rank == 0) {
        printf("IPC handles exchanged and opened successfully\n");
    }

    // Create stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));


    // Prepare batch copy parameters for AllGather
    // Each rank copies data from all other ranks to its gathered buffer
    std::vector<void*> src_ptrs(world_size);
    std::vector<void*> dst_ptrs(world_size);
    std::vector<size_t> sizes(world_size);
    
    for (int i = 0; i < world_size; i++) {
        src_ptrs[i] = remote_ptrs[i];
        dst_ptrs[i] = (char*)d_gathered_data + i * local_size;
        sizes[i] = local_size;
    }
    
    // Allocate device memory for batch copy parameters
    void** d_src_ptrs;
    void** d_dst_ptrs;
    size_t* d_sizes;
    size_t* d_failIdx;
    
    CUDA_CHECK(cudaMalloc(&d_src_ptrs, world_size * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_dst_ptrs, world_size * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_sizes, world_size * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_failIdx, sizeof(size_t)));
    
    CUDA_CHECK(cudaMemcpy(d_src_ptrs, src_ptrs.data(), 
                        world_size * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst_ptrs, dst_ptrs.data(), 
                        world_size * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, sizes.data(), 
                        world_size * sizeof(size_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
        // Perform AllGather using multiple cudaMemcpyAsync calls (NVIDIA equivalent
    // of hipMemcpyBatchAsync). Each rank copies from all ranks into its gathered buffer.
    for (int i = 0; i < world_size; i++) {
        void* src = remote_ptrs[i];
        void* dst = reinterpret_cast<char*>(d_gathered_data) + i * local_size;
        CUDA_CHECK(cudaMemcpyAsync(dst, src, local_size, cudaMemcpyDefault, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("cudaMemcpyAsync finished successfully\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cudaMemcpyAttributes attrs[1];
    attrs[0] = {};
    attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs[0].flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attrIdxs[] = {0,8};//The attributes specified in attrs[k] will be applied to copies starting from attrsIdxs[k] through attrsIdxs[k+1] - 1
    // Perform AllGather using cudaMemcpyBatchAsync
    CUDA_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), world_size, attrs, attrIdxs, 1, stream));
    // seg fault:
    //  CUDA_CHECK(cudaMemcpyBatchAsync(d_dst_ptrs, d_src_ptrs, d_sizes, world_size, attrs, attrIdxs, 1, stream));
    

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (world_rank == 0) {
        printf("AllGather completed using cudaMemcpyAsync (batch on stream)\n");
    }

    // Verify the gathered data by copying back to CPU
    std::vector<float> h_gathered_data(rows * cols * world_size);
    CUDA_CHECK(cudaMemcpy(h_gathered_data.data(), d_gathered_data, total_size,
                          cudaMemcpyDeviceToHost));

    int error_count = 0;
    for (int i = 0; i < rows * cols * world_size; i++) {
        int rank = i / (rows * cols);
        int local_idx = i % (rows * cols);
        float expected = rank * 1000.0f + static_cast<float>(local_idx);

        if (fabsf(h_gathered_data[i] - expected) > 1e-5f) {
            error_count++;
        }
    }

    if (error_count == 0) {
        printf("Rank %d: Verification PASSED - All data correct!\n", world_rank);
    } else {
        printf("Rank %d: Verification FAILED - %d errors found!\n",
               world_rank, error_count);
    }

    // Copy a sample of gathered data to host for inspection
    if (world_rank == 0) {
        printf("\nSample of gathered data (first 10 elements from each rank):\n");
        for (int i = 0; i < world_size; i++) {
            printf("Rank %d data: ", i);
            for (int j = 0; j < 10; j++) {
                printf("%.1f ", h_gathered_data[i * rows * cols + j]);
            }
            printf("...\n");
        }
    }

    // Cleanup
    for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
            CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptrs[i]));
        }
    }

    CUDA_CHECK(cudaFree(d_local_data));
    CUDA_CHECK(cudaFree(d_gathered_data));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("\nProgram completed successfully!\n");
    }

    MPI_Finalize();
    return 0;
}
