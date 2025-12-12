#include <stdio.h>
#include <stdlib.h>

// Serial version - runs on CPU
__global__ void addVectors(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (index< n) {//for each element in the array, add the elements

        c[index] = a[index] + b[index];//add the elements
    }
}



int main() {
    const int n = 5;//number of elements in the array
    int *h_a, *h_b, *h_c;  // host (CPU) arrays
    
    // Allocate host memory
    h_a = (int*)malloc(n * sizeof(int));//allocate memory for the array
    h_b = (int*)malloc(n * sizeof(int));//allocate memory for the array
    h_c = (int*)malloc(n * sizeof(int));//allocate memory for the array
    
    // Initialize arrays
    for (int i = 0; i < n; i++) {//for each element in the array, initialize the array
        h_a[i] = i;//initialize the array
        h_b[i] = i * 2;//initialize the array
    }
    
    // ========== SERIAL CODE SECTION ==========
    
    // Run serial version
    addVectorsSerial(h_a, h_b, h_c, n);//run the serial version
    
    // ========== END SERIAL CODE SECTION ==========
    
    // Print results
    printf("Serial Results:\n");//print the result
    for (int i = 0; i < n; i++) {//for each element in the array, print the result
        printf("c[%d] = %d\n", i, h_c[i]);//print the result
    }
    
    // Free host memory
    free(h_a);//free the memory for the array
    free(h_b);//free the memory for the array
    free(h_c);//free the memory for the array               
    
    return 0;//return 0
}
