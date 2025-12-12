#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel - runs on GPU
__global__ void addVectors(int *a, int *b, int *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;//index is the global thread index
    if (index < n) {//if the index is less than the number of elements in the array, then add the elements
        c[index] = a[index] + b[index];//add the elements
    }
}

// Serial version - runs on CPU
void addVectorsSerial(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {//for each element in the array, add the elements
        c[i] = a[i] + b[i];//add the elements
    }
}

int main() {
    const int n = 5;//number of elements in the array
    int *h_a, *h_b, *h_c;  // host (CPU) arrays
    int *d_a, *d_b, *d_c;  // device (GPU) arrays
    
    // Allocate host memory
    h_a = (int*)malloc(n * sizeof(int));//allocate memory for the array
    h_b = (int*)malloc(n * sizeof(int));//allocate memory for the array
    h_c = (int*)malloc(n * sizeof(int));//allocate memory for the array
    
    // Initialize arrays
    for (int i = 0; i < n; i++) {//for each element in the array, initialize the array
        h_a[i] = i;//initialize the array
        h_b[i] = i * 2;//initialize the array
    }
    
    // ========== CUDA CODE SECTION ==========
    
    // Allocate device memory
    cudaMalloc((void**)&d_a, n * sizeof(int));//allocate memory for the array
    cudaMalloc((void**)&d_b, n * sizeof(int));//allocate memory for the array
    cudaMalloc((void**)&d_c, n * sizeof(int));//allocate memory for the array
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);//copy the data from the host to the device
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);//copy the data from the host to the device               
    
    // Launch kernel
    int threadsPerBlock = 256;//number of threads per block
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;//number of blocks per grid
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);   
    
    // Copy result back from device to host
    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);//copy the data from the device to the host
    
    // Free device memory
    cudaFree(d_a);//free the memory for the array
    cudaFree(d_b);//free the memory for the array
    cudaFree(d_c);//free the memory for the array
    
    // ========== END CUDA CODE SECTION ==========
    
    // Print results
    printf("CUDA Results:\n");
    for (int i = 0; i < n; i++) {//for each element in the array, print the result
        printf("c[%d] = %d\n", i, h_c[i]);//print the result
    }
    
    // ========== SERIAL CODE SECTION ==========
    
    // Reset result array
    for (int i = 0; i < n; i++) {//for each element in the array, reset the array
        h_c[i] = 0; //reset the array
    }
    
    // Run serial version
    addVectorsSerial(h_a, h_b, h_c, n);//run the serial version
    
    // ========== END SERIAL CODE SECTION ==========
    
    // Print serial results
    printf("\nSerial Results:\n");//print the result
    for (int i = 0; i < n; i++) {//for each element in the array, print the result
        printf("c[%d] = %d\n", i, h_c[i]);//print the result
    }
    
    // Free host memory
    free(h_a);//free the memory for the array
    free(h_b);//free the memory for the array
    free(h_c);//free the memory for the array               
    
    return 0;//return 0
}

