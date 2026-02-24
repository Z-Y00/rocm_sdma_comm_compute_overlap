#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(cmd)                                                        \
    {                                                                          \
        cudaError_t error = cmd;                                               \
        if (error != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: '%s' (%d) at %s:%d\n",                \
                    cudaGetErrorString(error), error, __FILE__, __LINE__);     \
            MPI_Abort(MPI_COMM_WORLD, error);                                  \
        }                                                                      \
    }

#define CUBLAS_CHECK(cmd)                                                      \
    {                                                                          \
        cublasStatus_t status = cmd;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error: %d at %s:%d\n",                     \
                    static_cast<int>(status), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));               \
        }                                                                      \
    }

#define MPI_CHECK(cmd)                                                         \
    {                                                                          \
        int error = cmd;                                                       \
        if (error != MPI_SUCCESS) {                                            \
            fprintf(stderr, "MPI error: %d at %s:%d\n",                        \
                    error, __FILE__, __LINE__);                                \
            MPI_Abort(MPI_COMM_WORLD, error);                                  \
        }                                                                      \
    }

static void enqueue_large_gemm(cublasHandle_t handle,
                               float* d_a,
                               float* d_b,
                               float* d_c,
                               int m,
                               int n,
                               int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        d_a,
        m,
        d_b,
        k,
        &beta,
        d_c,
        m));
}

static float run_ordering_case(int rank,
                               cudaStream_t stream,
                               cublasHandle_t cublas_handle,
                               float* d_a,
                               float* d_b,
                               float* d_c,
                               void* const* dst_ptrs,
                               const void* const* src_ptrs,
                               const size_t* sizes,
                               size_t num_copies,
                               cudaMemcpyAttributes* attrs,
                               size_t* attr_idxs,
                               bool gemm_first,
                               int m,
                               int n,
                               int k) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    if (gemm_first) {
        enqueue_large_gemm(cublas_handle, d_a, d_b, d_c, m, n, k);
        CUDA_CHECK(cudaMemcpyBatchAsync(
            dst_ptrs, src_ptrs, sizes, num_copies, attrs, attr_idxs, 1, stream));
    } else {
        CUDA_CHECK(cudaMemcpyBatchAsync(
            dst_ptrs, src_ptrs, sizes, num_copies, attrs, attr_idxs, 1, stream));
        enqueue_large_gemm(cublas_handle, d_a, d_b, d_c, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (rank == 0) {
        printf("%s elapsed: %.3f ms\n",
               gemm_first ? "Case 1 (GEMM -> cudaMemcpyBatchAsync)" :
                            "Case 2 (cudaMemcpyBatchAsync -> GEMM)",
               elapsed_ms);
    }
    return elapsed_ms;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank = 0;
    int world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device found.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    // Keep defaults simple but configurable:
    // argv[1]=m, argv[2]=n, argv[3]=k, argv[4]=copy_bytes
    int m = 4096;
    int n = 4096;
    int k = 4096;
    size_t copy_bytes = 64ULL * 1024ULL * 1024ULL;  // 64 MiB per copy
    if (argc > 1) m = std::atoi(argv[1]);
    if (argc > 2) n = std::atoi(argv[2]);
    if (argc > 3) k = std::atoi(argv[3]);
    if (argc > 4) copy_bytes = static_cast<size_t>(std::strtoull(argv[4], nullptr, 10));

    if (rank == 0) {
        printf("Running same-stream ordering test on %d rank(s).\n", world_size);
        printf("Device count=%d, rank0 device=%d, GEMM=(%d,%d,%d), copy_bytes=%zu\n",
               device_count, device_id, m, n, k, copy_bytes);
        printf("Both cases use cudaMemcpyFlagPreferOverlapWithCompute.\n");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    // GEMM buffers.
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(m) * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(k) * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_a, 1, static_cast<size_t>(m) * k * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_b, 2, static_cast<size_t>(k) * n * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_c, 0, static_cast<size_t>(m) * n * sizeof(float), stream));

    // Batch memcpy buffers (pinned host + device).
    float* h_src = nullptr;
    float* h_dst = nullptr;
    void* d_buf1 = nullptr;
    void* d_buf2 = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_src, copy_bytes));
    CUDA_CHECK(cudaMallocHost(&h_dst, copy_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf1, copy_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf2, copy_bytes));

    const size_t num_float = copy_bytes / sizeof(float);
    for (size_t i = 0; i < num_float; i++) {
        h_src[i] = static_cast<float>((i + rank) % 1024);
    }

    const size_t num_copies = 3;
    const void* src_ptrs[num_copies] = {h_src, d_buf1, d_buf2};
    void* dst_ptrs[num_copies] = {d_buf1, d_buf2, h_dst};
    size_t sizes[num_copies] = {copy_bytes, copy_bytes, copy_bytes};

    cudaMemcpyAttributes attrs[1] = {};
    attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs[0].flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idxs[] = {0, num_copies};

    // Warm up the stream once so first-use overhead is not part of case timing.
    enqueue_large_gemm(cublas_handle, d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaMemcpyBatchAsync(
        dst_ptrs, src_ptrs, sizes, num_copies, attrs, attr_idxs, 1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    run_ordering_case(rank,
                      stream,
                      cublas_handle,
                      d_a,
                      d_b,
                      d_c,
                      dst_ptrs,
                      src_ptrs,
                      sizes,
                      num_copies,
                      attrs,
                      attr_idxs,
                      true,
                      m,
                      n,
                      k);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    run_ordering_case(rank,
                      stream,
                      cublas_handle,
                      d_a,
                      d_b,
                      d_c,
                      dst_ptrs,
                      src_ptrs,
                      sizes,
                      num_copies,
                      attrs,
                      attr_idxs,
                      false,
                      m,
                      n,
                      k);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Quick touch to ensure outputs were materialized.
    float c0 = 0.0f;
    CUDA_CHECK(cudaMemcpy(&c0, d_c, sizeof(float), cudaMemcpyDeviceToHost));
    if (rank == 0) {
        printf("Sanity values: GEMM C[0]=%.4f, memcpy h_dst[0]=%.4f\n", c0, h_dst[0]);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaFree(d_buf2));
    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_dst));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}
