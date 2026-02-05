#include <hip/hip_runtime.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd)                                                         \
    {                                                                          \
        hipError_t error = cmd;                                                \
        if (error != hipSuccess) {                                             \
            fprintf(stderr, "Error: '%s'(%d) at %s:%d\n",                      \
                    hipGetErrorString(error), error, __FILE__, __LINE__);      \
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
    HIP_CHECK(hipGetDeviceCount(&device_count));
    
    if (device_count < 8) {
        if (world_rank == 0) {
            fprintf(stderr, "This program requires at least 8 GPUs, found %d\n", 
                    device_count);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Each process uses its rank as device ID
    HIP_CHECK(hipSetDevice(world_rank));
    
    if (world_rank == 0) {
        printf("Running HIP IPC AllGather with 8 processes on 8 GPUs\n");
    }
    
    // Define 2D array dimensions for each rank
    const int rows = 1024;
    const int cols = 1024;
    const size_t local_size = rows * cols * sizeof(float);
    const size_t total_size = local_size * world_size;
    
    // Allocate local device memory for this rank's data
    float* d_local_data;
    HIP_CHECK(hipMalloc(&d_local_data, local_size));
    
    // Initialize local data on CPU
    std::vector<float> h_local_data(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        // Each rank initializes its data with rank-specific values
        h_local_data[i] = world_rank * 1000.0f + i;
    }
    
    // Copy initialized data to GPU
    HIP_CHECK(hipMemcpy(d_local_data, h_local_data.data(), local_size, 
                        hipMemcpyHostToDevice));
    
    // Create IPC handle for local data
    hipIpcMemHandle_t local_ipc_handle;
    HIP_CHECK(hipIpcGetMemHandle(&local_ipc_handle, d_local_data));
    
    // Gather all IPC handles to all processes
    std::vector<hipIpcMemHandle_t> all_ipc_handles(world_size);
    MPI_CHECK(MPI_Allgather(&local_ipc_handle, sizeof(hipIpcMemHandle_t), 
                            MPI_BYTE, all_ipc_handles.data(), 
                            sizeof(hipIpcMemHandle_t), MPI_BYTE, 
                            MPI_COMM_WORLD));
    
    // Allocate memory for gathered data (all ranks' data)
    float* d_gathered_data;
    HIP_CHECK(hipMalloc(&d_gathered_data, total_size));
    
    // Open IPC handles to access remote memory
    std::vector<void*> remote_ptrs(world_size);
    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) {
            // Use local pointer for own data
            remote_ptrs[i] = d_local_data;
        } else {
            // Open IPC handle for remote rank's memory
            HIP_CHECK(hipIpcOpenMemHandle(&remote_ptrs[i], 
                                          all_ipc_handles[i],
                                          hipIpcMemLazyEnablePeerAccess));
        }
    }
    
    if (world_rank == 0) {
        printf("IPC handles exchanged and opened successfully\n");
    }
    
    // Create stream for async operations
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
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
    
    HIP_CHECK(hipMalloc(&d_src_ptrs, world_size * sizeof(void*)));
    HIP_CHECK(hipMalloc(&d_dst_ptrs, world_size * sizeof(void*)));
    HIP_CHECK(hipMalloc(&d_sizes, world_size * sizeof(size_t)));
    HIP_CHECK(hipMalloc(&d_failIdx, sizeof(size_t)));
    
    HIP_CHECK(hipMemcpy(d_src_ptrs, src_ptrs.data(), 
                        world_size * sizeof(void*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dst_ptrs, dst_ptrs.data(), 
                        world_size * sizeof(void*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sizes, sizes.data(), 
                        world_size * sizeof(size_t), hipMemcpyHostToDevice));
    
    // Perform AllGather using hipMemcpyBatchAsync
    HIP_CHECK(hipMemcpyBatchAsync(d_dst_ptrs, d_src_ptrs, d_sizes, 
                                  world_size, nullptr, nullptr, 0, d_failIdx, stream));
    
    HIP_CHECK(hipStreamSynchronize(stream));
    
    if (world_rank == 0) {
        printf("AllGather completed using hipMemcpyBatchAsync\n");
    }
    
    // Verify the gathered data by copying back to CPU
    std::vector<float> h_gathered_data(rows * cols * world_size);
    HIP_CHECK(hipMemcpy(h_gathered_data.data(), d_gathered_data, total_size,
                        hipMemcpyDeviceToHost));
    
    int error_count = 0;
    for (int i = 0; i < rows * cols * world_size; i++) {
        int rank = i / (rows * cols);
        int local_idx = i % (rows * cols);
        float expected = rank * 1000.0f + local_idx;
        
        if (fabsf(h_gathered_data[i] - expected) > 1e-5) {
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
            HIP_CHECK(hipIpcCloseMemHandle(remote_ptrs[i]));
        }
    }
    
    HIP_CHECK(hipFree(d_local_data));
    HIP_CHECK(hipFree(d_gathered_data));
    HIP_CHECK(hipFree(d_src_ptrs));
    HIP_CHECK(hipFree(d_dst_ptrs));
    HIP_CHECK(hipFree(d_sizes));
    HIP_CHECK(hipFree(d_failIdx));
    HIP_CHECK(hipStreamDestroy(stream));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        printf("\nProgram completed successfully!\n");
    }
    
    MPI_Finalize();
    return 0;
}
