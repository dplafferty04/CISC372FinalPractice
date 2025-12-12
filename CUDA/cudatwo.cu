#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <math.h>

#define TILE_SIZE 16
#define WARP_SIZE 32

// ============================================================================
// 1. SYNCHRONIZATION EXAMPLES
// ============================================================================

/*
 * __syncthreads() - Synchronization within a block
 * 
 * Synchronizes all threads within a block. Waits until all threads in the block
 * reach this point. Essential when using shared memory to ensure data is
 * written before reading. Only synchronizes threads in the SAME block
 * (not across blocks).
 * 
 * Usage: __syncthreads();
 */
__global__ void syncExample(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // All threads in block write to shared memory
        __shared__ int shared_data[256];
        shared_data[threadIdx.x] = data[idx];
        
        // Synchronize all threads in the block before reading
        __syncthreads();
        
        // Now safe to read from shared memory written by other threads
        if (threadIdx.x < blockDim.x - 1) {
            data[idx] = shared_data[threadIdx.x] + shared_data[threadIdx.x + 1];
        }
    }
}

// ============================================================================
// 2. MATRIX MULTIPLICATION EXAMPLES
// ============================================================================

/*
 * 2D Matrix Multiplication - Simple version
 * 
 * Demonstrates 2D kernel launch using dim3 for grid and block dimensions.
 * Uses 2D thread indexing:
 *   - row = blockIdx.y * blockDim.y + threadIdx.y
 *   - col = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * Each thread computes one element of the result matrix C.
 */
__global__ void matrixMultiply2D(float *A, float *B, float *C, 
                                  int widthA, int widthB) {
    // 2D thread indexing
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < widthA && col < widthB) {
        float sum = 0.0f;
        for (int k = 0; k < widthA; k++) {
            sum += A[row * widthA + k] * B[k * widthB + col];
        }
        C[row * widthB + col] = sum;
    }
}

/*
 * Tiled Matrix Multiplication with Shared Memory
 * 
 * Divides matrices into smaller tiles (typically 16x16 or 32x32) and loads
 * them into shared memory for faster access. This reduces global memory
 * accesses significantly.
 * 
 * Key steps:
 *   1. Load tile from global to shared memory
 *   2. __syncthreads() to ensure all threads loaded
 *   3. Compute partial products using shared memory
 *   4. __syncthreads() before loading next tile
 * 
 * Shared memory is declared with: __shared__ type name[size];
 * - Shared among all threads in the same block
 * - Much faster than global memory
 * - Limited size per block (typically 48KB)
 */
__global__ void matrixMultiplyTiled(float *A, float *B, float *C,
                                     int widthA, int widthB) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (widthA + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile from A into shared memory
        int aRow = row;
        int aCol = tile * TILE_SIZE + threadIdx.x;
        if (aRow < widthA && aCol < widthA) {
            tileA[threadIdx.y][threadIdx.x] = A[aRow * widthA + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        int bRow = tile * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < widthA && bCol < widthB) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * widthB + bCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < widthA && col < widthB) {
        C[row * widthB + col] = sum;
    }
}

// ============================================================================
// 3. MANUAL REDUCTION ALGORITHMS
// ============================================================================

/*
 * Tree-based Parallel Reduction
 * 
 * Reduces an array to a single value (sum, max, min, etc.) using a tree-based
 * approach. Each level halves the active threads.
 * 
 * Pattern:
 *   - Load data into shared memory
 *   - Iteratively reduce: s >>= 1 (divide stride by 2)
 *   - Each thread adds its value with value at (tid + s)
 *   - Continue until stride becomes 0
 *   - Final result in shared_data[0]
 * 
 * Requires __syncthreads() after each reduction step.
 */
__global__ void reductionTree(int *input, int *output, int n) {
    __shared__ int sdata[512];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Tree-based reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/*
 * Reduction using Atomic Operations
 * 
 * Atomic operations ensure thread-safe operations on shared/global memory.
 * Common atomic functions:
 *   - atomicAdd(&var, value)    - Add value to variable
 *   - atomicSub(&var, value)    - Subtract value from variable
 *   - atomicExch(&var, value)   - Exchange value
 *   - atomicMin(&var, value)    - Minimum
 *   - atomicMax(&var, value)    - Maximum
 * 
 * Slower than regular operations but thread-safe. Use when multiple threads
 * update the same memory location.
 */
__global__ void reductionAtomic(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Atomic add to global memory
        atomicAdd(output, input[idx]);
    }
}

// ============================================================================
// 4. MULTIPLE DIMENSIONS (2D/3D)
// ============================================================================

/*
 * 2D Kernel with dim3
 * 
 * dim3 is a CUDA structure with .x, .y, .z members for multi-dimensional
 * grids and blocks. Default values: .x = specified, .y = 1, .z = 1
 * 
 * Access dimensions:
 *   - threadIdx.x, threadIdx.y, threadIdx.z - thread index within block
 *   - blockIdx.x, blockIdx.y, blockIdx.z - block index within grid
 *   - blockDim.x, blockDim.y, blockDim.z - block dimensions
 *   - gridDim.x, gridDim.y, gridDim.z - grid dimensions
 * 
 * 2D indexing:
 *   int x = blockIdx.x * blockDim.x + threadIdx.x;
 *   int y = blockIdx.y * blockDim.y + threadIdx.y;
 *   int idx = y * width + x;
 */
__global__ void kernel2D(int *data, int width, int height) {
    // Using .x and .y dimensions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = x + y;
    }
}

/*
 * 3D Kernel with dim3
 * 
 * Extends 2D concepts to three dimensions. Useful for 3D data structures
 * like volumes, 3D arrays, etc.
 * 
 * 3D indexing:
 *   int x = blockIdx.x * blockDim.x + threadIdx.x;
 *   int y = blockIdx.y * blockDim.y + threadIdx.y;
 *   int z = blockIdx.z * blockDim.z + threadIdx.z;
 *   int idx = z * width * height + y * width + x;
 */
__global__ void kernel3D(int *data, int width, int height, int depth) {
    // Using .x, .y, and .z dimensions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x;
        data[idx] = x + y + z;
    }
}

// ============================================================================
// 5. LIMITATIONS DEMONSTRATION
// ============================================================================

/*
 * CUDA Limitations Demonstration
 * 
 * Key limitations to be aware of:
 *   - Max threads per block: typically 1024 (check with cudaDeviceProp.maxThreadsPerBlock)
 *   - Warp size: always 32 threads per warp
 *   - Shared memory: limited per block (typically 48KB or 96KB)
 *   - Register limits: each thread has limited registers
 *   - Max grid dimensions: 2D: 65535 x 65535, 3D: 65535 x 65535 x 65535
 * 
 * Warp-aware programming:
 *   int warp_id = threadIdx.x / 32;
 *   int lane_id = threadIdx.x % 32;
 * 
 * Threads in a warp execute in lockstep (SIMT). Divergence (different paths)
 * degrades performance.
 */
__global__ void checkLimits() {
    // Max threads per block: typically 1024
    // Warp size: 32 threads
    // Shared memory: limited per block (varies by GPU)
    
    __shared__ int shared_array[1024]; // Shared memory example
    
    int tid = threadIdx.x;
    
    // Warp-aware programming
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Example: warp-level operations
    if (lane_id == 0) {
        shared_array[warp_id] = warp_id;
    }
    __syncthreads();
}

// ============================================================================
// 6. API LIBRARIES EXAMPLES
// ============================================================================

/*
 * cuBLAS (CUDA Basic Linear Algebra Subroutines)
 * 
 * High-performance linear algebra library for matrix operations.
 * Steps:
 *   1. Create handle: cublasCreate(&handle)
 *   2. Set matrix data on device
 *   3. Call function (e.g., cublasSgemm for matrix multiply)
 *   4. Destroy handle: cublasDestroy(handle)
 * 
 * Compile with: -lcublas
 */
void cublasExample() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    int n = 4;
    float *h_A = (float*)malloc(n * n * sizeof(float));
    float *h_B = (float*)malloc(n * n * sizeof(float));
    float *h_C = (float*)malloc(n * n * sizeof(float));
    
    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));
    
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // cuBLAS matrix multiplication: C = alpha * A * B + beta * C
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha, d_A, n,
                d_B, n,
                &beta, d_C, n);
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("cuBLAS Result (first element): %.2f\n", h_C[0]);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cublasDestroy(handle);
}

/*
 * cuRAND (CUDA Random Number Generation)
 * 
 * Generates random numbers directly on GPU.
 * Steps:
 *   1. Create generator: curandCreateGenerator(&gen, type)
 *   2. Set seed: curandSetPseudoRandomGeneratorSeed(gen, seed)
 *   3. Generate: curandGenerateUniform(gen, d_data, n)
 *   4. Destroy: curandDestroyGenerator(gen)
 * 
 * Compile with: -lcurand
 */
void curandExample() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    int n = 1000;
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    // Generate random numbers
    curandGenerateUniform(gen, d_data, n);
    
    float *h_data = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("cuRAND Sample (first 5): ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", h_data[i]);
    }
    printf("\n");
    
    cudaFree(d_data);
    free(h_data);
    curandDestroyGenerator(gen);
}

/*
 * Stream Synchronization
 * 
 * cudaStreamSynchronize(stream) synchronizes a specific CUDA stream.
 * Allows asynchronous execution across multiple streams. Only waits for
 * operations in the specified stream.
 * 
 * Usage: cudaStreamSynchronize(stream1);
 * 
 * Also see: cudaDeviceSynchronize() - waits for all device operations
 */
void streamSyncExample() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    int n = 1000;
    int *d_data1, *d_data2;
    cudaMalloc(&d_data1, n * sizeof(int));
    cudaMalloc(&d_data2, n * sizeof(int));
    
    // Launch kernels in different streams
    syncExample<<<1, 256, 0, stream1>>>(d_data1, n);
    syncExample<<<1, 256, 0, stream2>>>(d_data2, n);
    
    // Synchronize specific stream
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// ============================================================================
// MAIN FUNCTION - DEMONSTRATION
// ============================================================================

int main() {
    printf("=== CUDA Parallelization Examples ===\n\n");
    
    // 1. Synchronization example
    printf("1. Synchronization Example:\n");
    int n = 100;
    int *h_data = (int*)malloc(n * sizeof(int));
    int *d_data;
    
    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }
    
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    
    syncExample<<<1, 256>>>(d_data, n);
    cudaDeviceSynchronize(); // Host waits for device
    
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("   Result[0] = %d\n", h_data[0]);
    
    cudaFree(d_data);
    free(h_data);
    
    // 2. 2D Matrix Multiplication
    printf("\n2. 2D Matrix Multiplication:\n");
    int width = 4, height = 4;
    float *h_A = (float*)malloc(width * height * sizeof(float));
    float *h_B = (float*)malloc(width * height * sizeof(float));
    float *h_C = (float*)malloc(width * height * sizeof(float));
    
    for (int i = 0; i < width * height; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, width * height * sizeof(float));
    cudaMalloc(&d_B, width * height * sizeof(float));
    cudaMalloc(&d_C, width * height * sizeof(float));
    
    cudaMemcpy(d_A, h_A, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    // 2D grid and block dimensions
    dim3 blockSize(4, 4);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    matrixMultiply2D<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, width);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    printf("   C[0] = %.2f\n", h_C[0]);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    // 3. Tiled Matrix Multiplication
    printf("\n3. Tiled Matrix Multiplication:\n");
    int tile_width = 8;
    h_A = (float*)malloc(tile_width * tile_width * sizeof(float));
    h_B = (float*)malloc(tile_width * tile_width * sizeof(float));
    h_C = (float*)malloc(tile_width * tile_width * sizeof(float));
    
    for (int i = 0; i < tile_width * tile_width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    
    cudaMalloc(&d_A, tile_width * tile_width * sizeof(float));
    cudaMalloc(&d_B, tile_width * tile_width * sizeof(float));
    cudaMalloc(&d_C, tile_width * tile_width * sizeof(float));
    
    cudaMemcpy(d_A, h_A, tile_width * tile_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, tile_width * tile_width * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 tileBlock(TILE_SIZE, TILE_SIZE);
    dim3 tileGrid((tile_width + TILE_SIZE - 1) / TILE_SIZE,
                  (tile_width + TILE_SIZE - 1) / TILE_SIZE);
    
    matrixMultiplyTiled<<<tileGrid, tileBlock>>>(d_A, d_B, d_C, tile_width, tile_width);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, tile_width * tile_width * sizeof(float), cudaMemcpyDeviceToHost);
    printf("   Tiled C[0] = %.2f\n", h_C[0]);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    // 4. Reduction Example
    printf("\n4. Tree-based Reduction:\n");
    int reduce_n = 512;
    int *h_input = (int*)malloc(reduce_n * sizeof(int));
    int *h_output = (int*)malloc(sizeof(int));
    int *d_input, *d_output;
    
    for (int i = 0; i < reduce_n; i++) {
        h_input[i] = 1;
    }
    
    cudaMalloc(&d_input, reduce_n * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));
    
    cudaMemcpy(d_input, h_input, reduce_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(int));
    
    reductionTree<<<1, 512>>>(d_input, d_output, reduce_n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("   Sum = %d\n", h_output[0]);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    // 5. 3D Kernel Example
    printf("\n5. 3D Kernel Example:\n");
    int dim3_width = 4, dim3_height = 4, dim3_depth = 4;
    int *h_3d = (int*)malloc(dim3_width * dim3_height * dim3_depth * sizeof(int));
    int *d_3d;
    
    cudaMalloc(&d_3d, dim3_width * dim3_height * dim3_depth * sizeof(int));
    
    dim3 block3D(2, 2, 2);
    dim3 grid3D((dim3_width + block3D.x - 1) / block3D.x,
                (dim3_height + block3D.y - 1) / block3D.y,
                (dim3_depth + block3D.z - 1) / block3D.z);
    
    kernel3D<<<grid3D, block3D>>>(d_3d, dim3_width, dim3_height, dim3_depth);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_3d, d_3d, dim3_width * dim3_height * dim3_depth * sizeof(int),
               cudaMemcpyDeviceToHost);
    printf("   3D data[0] = %d\n", h_3d[0]);
    
    cudaFree(d_3d);
    free(h_3d);
    
    // 6. Limitations check
    printf("\n6. CUDA Limitations:\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("   Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("   Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("   Warp size: %d\n", prop.warpSize);
    printf("   Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    
    // 7. API Libraries (commented out - requires linking)
    printf("\n7. API Libraries:\n");
    printf("   (cuBLAS and cuRAND examples are in code but require linking)\n");
    printf("   Compile with: nvcc -lcublas -lcurand cudatwo.cu\n");
    
    printf("\n=== Examples Complete ===\n");
    
    return 0;
}

/*
 * ============================================================================
 * CUDA PARALLELIZATION REFERENCE GUIDE
 * ============================================================================
 * 
 * This is a comprehensive reference guide for CUDA parallelization concepts.
 * Use this as a quick lookup for key CUDA programming topics.
 * 
 * ============================================================================
 * 1. SYNCHRONIZATION
 * ============================================================================
 * 
 * __syncthreads()
 *   - Synchronizes all threads within a block
 *   - Waits until all threads in the block reach this point
 *   - Essential when using shared memory to ensure data is written before reading
 *   - Only synchronizes threads in the SAME block (not across blocks)
 *   - Usage: __syncthreads();
 * 
 * cudaDeviceSynchronize()
 *   - Host function that waits for all device operations to complete
 *   - Blocks CPU execution until GPU finishes all kernels and memory operations
 *   - Useful for timing and ensuring data is ready before CPU access
 *   - Usage: cudaDeviceSynchronize();
 * 
 * cudaStreamSynchronize(stream)
 *   - Synchronizes a specific CUDA stream
 *   - Allows asynchronous execution across multiple streams
 *   - Only waits for operations in the specified stream
 *   - Usage: cudaStreamSynchronize(stream1);
 * 
 * ============================================================================
 * 2. MATRIX MULTIPLICATION IN CUDA
 * ============================================================================
 * 
 * 2D Kernel Launch
 *   - Use dim3 for 2D/3D grid and block dimensions
 *   - Access thread indices: threadIdx.x, threadIdx.y
 *   - Access block indices: blockIdx.x, blockIdx.y
 *   - Calculate global indices:
 *       row = blockIdx.y * blockDim.y + threadIdx.y;
 *       col = blockIdx.x * blockDim.x + threadIdx.x;
 * 
 * Tiled Matrix Multiplication
 *   - Divides matrices into smaller tiles (typically 16x16 or 32x32)
 *   - Loads tiles into shared memory for faster access
 *   - Reduces global memory accesses
 *   - Requires __syncthreads() after loading each tile
 *   - Key steps:
 *       1. Load tile from global to shared memory
 *       2. __syncthreads() to ensure all threads loaded
 *       3. Compute partial products using shared memory
 *       4. __syncthreads() before loading next tile
 * 
 * Shared Memory Usage
 *   - Declared with: __shared__ type name[size];
 *   - Shared among all threads in the same block
 *   - Much faster than global memory
 *   - Limited size per block (typically 48KB)
 *   - Must synchronize before reading data written by other threads
 * 
 * ============================================================================
 * 3. MANUAL REDUCTION ALGORITHMS
 * ============================================================================
 * 
 * Parallel Reduction Patterns
 *   - Reduces an array to a single value (sum, max, min, etc.)
 *   - Tree-based approach: each level halves the active threads
 *   - Starts with all threads, reduces by half each iteration
 * 
 * Tree-based Reduction
 *   - Load data into shared memory
 *   - Iteratively reduce: s >>= 1 (divide stride by 2)
 *   - Each thread adds its value with value at (tid + s)
 *   - Continue until stride becomes 0
 *   - Final result in shared_data[0]
 *   - Example pattern:
 *       for (int s = blockDim.x / 2; s > 0; s >>= 1) {
 *           if (tid < s) {
 *               sdata[tid] += sdata[tid + s];
 *           }
 *           __syncthreads();
 *       }
 * 
 * Atomic Operations
 *   - Ensures thread-safe operations on shared/global memory
 *   - Common functions:
 *       atomicAdd(&var, value)    - Add value to variable
 *       atomicSub(&var, value)    - Subtract value from variable
 *       atomicExch(&var, value)   - Exchange value
 *       atomicMin(&var, value)    - Minimum
 *       atomicMax(&var, value)    - Maximum
 *   - Slower than regular operations but thread-safe
 *   - Use when multiple threads update same memory location
 * 
 * ============================================================================
 * 4. API LIBRARIES
 * ============================================================================
 * 
 * cuBLAS (CUDA Basic Linear Algebra Subroutines)
 *   - High-performance linear algebra library
 *   - Matrix multiplication, vector operations, etc.
 *   - Steps:
 *       1. Create handle: cublasCreate(&handle)
 *       2. Set matrix data on device
 *       3. Call function (e.g., cublasSgemm for matrix multiply)
 *       4. Destroy handle: cublasDestroy(handle)
 *   - Compile with: -lcublas
 * 
 * cuRAND (CUDA Random Number Generation)
 *   - Generates random numbers on GPU
 *   - Steps:
 *       1. Create generator: curandCreateGenerator(&gen, type)
 *       2. Set seed: curandSetPseudoRandomGeneratorSeed(gen, seed)
 *       3. Generate: curandGenerateUniform(gen, d_data, n)
 *       4. Destroy: curandDestroyGenerator(gen)
 *   - Compile with: -lcurand
 * 
 * Other CUDA Libraries
 *   - cuFFT: Fast Fourier Transform
 *   - cuSPARSE: Sparse matrix operations
 *   - cuDNN: Deep neural network primitives
 *   - Thrust: C++ template library (like STL for GPU)
 * 
 * ============================================================================
 * 5. MULTIPLE DIMENSIONS (2D/3D)
 * ============================================================================
 * 
 * dim3 for 2D/3D Grids and Blocks
 *   - dim3 is a CUDA structure with .x, .y, .z members
 *   - Default values: .x = specified, .y = 1, .z = 1
 *   - Example:
 *       dim3 blockSize(16, 16);        // 2D: 16x16 = 256 threads
 *       dim3 gridSize(4, 4);            // 2D: 4x4 = 16 blocks
 *       dim3 block3D(8, 8, 8);         // 3D: 8x8x8 = 512 threads
 * 
 * Using .x, .y, .z Dimensions
 *   - threadIdx.x, threadIdx.y, threadIdx.z - thread index within block
 *   - blockIdx.x, blockIdx.y, blockIdx.z - block index within grid
 *   - blockDim.x, blockDim.y, blockDim.z - block dimensions
 *   - gridDim.x, gridDim.y, gridDim.z - grid dimensions
 * 
 * 2D/3D Thread Indexing
 *   - 2D: 
 *       int x = blockIdx.x * blockDim.x + threadIdx.x;
 *       int y = blockIdx.y * blockDim.y + threadIdx.y;
 *       int idx = y * width + x;
 * 
 *   - 3D:
 *       int x = blockIdx.x * blockDim.x + threadIdx.x;
 *       int y = blockIdx.y * blockDim.y + threadIdx.y;
 *       int z = blockIdx.z * blockDim.z + threadIdx.z;
 *       int idx = z * width * height + y * width + x;
 * 
 * ============================================================================
 * 6. LIMITATIONS OF THREADS WITHIN A BLOCK
 * ============================================================================
 * 
 * Max Threads Per Block
 *   - Typically 1024 threads per block (varies by GPU)
 *   - Check with: cudaDeviceProp.maxThreadsPerBlock
 *   - Common configurations: 256, 512, 1024
 *   - Must be multiple of warp size (32)
 * 
 * Shared Memory Limits
 *   - Limited per block (typically 48KB or 96KB)
 *   - Check with: cudaDeviceProp.sharedMemPerBlock
 *   - Shared memory is faster than global memory
 *   - Trade-off: more shared memory = fewer blocks per SM
 * 
 * Register Limits
 *   - Each thread has limited registers
 *   - Too many registers = fewer threads per block
 *   - Check with: cudaDeviceProp.regsPerBlock
 *   - Can limit register usage with: __launch_bounds__(maxThreads, minBlocks)
 * 
 * Warp Size
 *   - Always 32 threads per warp
 *   - Threads in a warp execute in lockstep (SIMT)
 *   - Warp-aware programming:
 *       int warp_id = threadIdx.x / 32;
 *       int lane_id = threadIdx.x % 32;
 *   - Divergence: if threads in warp take different paths, performance degrades
 * 
 * Other Important Limits
 *   - Max threads per multiprocessor: varies (check maxThreadsPerMultiProcessor)
 *   - Max blocks per multiprocessor: varies by GPU
 *   - Max grid dimensions: 
 *       - 1D: 2^31 - 1
 *       - 2D: 65535 x 65535
 *       - 3D: 65535 x 65535 x 65535
 * 
 * ============================================================================
 * KEY CUDA MEMORY TYPES
 * ============================================================================
 * 
 * Global Memory (__device__)
 *   - Accessible by all threads, all blocks
 *   - Large but slow
 *   - Allocated with: cudaMalloc()
 *   - Lifetime: until cudaFree()
 * 
 * Shared Memory (__shared__)
 *   - Shared within a block only
 *   - Fast (on-chip)
 *   - Limited size per block
 *   - Lifetime: block execution
 * 
 * Registers
 *   - Private to each thread
 *   - Fastest memory
 *   - Limited per thread
 *   - Lifetime: thread execution
 * 
 * Constant Memory (__constant__)
 *   - Read-only, cached
 *   - Fast for read-only data
 *   - Limited size (64KB)
 * 
 * ============================================================================
 * COMPILATION NOTES
 * ============================================================================
 * 
 * Basic compilation:
 *   nvcc cudatwo.cu -o cudatwo
 * 
 * With libraries:
 *   nvcc -lcublas -lcurand cudatwo.cu -o cudatwo
 * 
 * With optimization:
 *   nvcc -O3 -arch=sm_XX cudatwo.cu -o cudatwo
 *   (Replace XX with your GPU compute capability, e.g., sm_75 for Turing)
 * 
 * ============================================================================
 */
