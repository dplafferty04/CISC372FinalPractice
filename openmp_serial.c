#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// SERIAL VERSION - No Parallelization
// ============================================================================
// This file shows the serial (sequential) versions of the OpenMP examples.
// All loops run sequentially on a single CPU thread.
// ============================================================================

// ============================================================================
// 1. SERIAL VERSION - CPU Loop
// ============================================================================

void serial_example(int *a, int n) {
    // Serial loop - runs sequentially on one CPU thread
    for (int i = 0; i < n; i++) {
        a[i] = i * 2;
    }
}

// ============================================================================
// 2. SERIAL VERSION - Basic Function
// ============================================================================

void basic_example() {
    // Code runs on CPU sequentially
    printf("Hello from CPU!\n");
}

// ============================================================================
// 3. SERIAL VERSION - Vector Addition
// ============================================================================

void vec_add(float *a, float *b, float *c, int n) {
    // Serial loop - processes elements one at a time
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// 4. SERIAL VERSION - Reduction (Sum)
// ============================================================================

// Serial reduction - no data race, just sequential accumulation
void serial_reduction(double *a, int n) {
    double sum = 0.0;
    // Serial loop - no parallelization, so no data race
    for (int i = 0; i < n; i++) {
        sum += a[i];  // Safe in serial - only one thread
    }
    printf("Sum: %.2f\n", sum);
}

// ============================================================================
// 5. SERIAL VERSION - Nested Loops
// ============================================================================

void nested_loops_example(float *A, float *B, int n, int m) {
    // Serial nested loops - processes all iterations sequentially
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int idx = i * m + j;
            B[idx] = A[idx] * 2.0f;
        }
    }
}

// ============================================================================
// 6. SERIAL VERSION - Dot Product
// ============================================================================

double dot_product(double *a, double *b, int n) {
    double sum = 0.0;
    
    // Serial loop - computes dot product sequentially
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

// ============================================================================
// 7. MAIN FUNCTION - Complete Working Example
// ============================================================================

int main() {
    const int n = 1000;
    
    // Allocate and initialize arrays
    float *a = (float*)malloc(n * sizeof(float));
    float *b = (float*)malloc(n * sizeof(float));
    float *c = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2.0f;
    }
    
    // Example: Vector addition (serial)
    vec_add(a, b, c, n);
    
    // Print first few results
    printf("Vector addition results (first 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %.1f\n", i, c[i]);
    }
    
    // Example: Dot product (serial)
    double *da = (double*)malloc(n * sizeof(double));
    double *db = (double*)malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        da[i] = i;
        db[i] = i;
    }
    
    double result = dot_product(da, db, n);
    printf("\nDot product result: %.2f\n", result);
    
    // Example: Reduction (serial)
    double *reduction_array = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        reduction_array[i] = i;
    }
    serial_reduction(reduction_array, n);
    
    // Cleanup
    free(a);
    free(b);
    free(c);
    free(da);
    free(db);
    free(reduction_array);
    
    return 0;
}

// ============================================================================
// SERIAL vs PARALLEL COMPARISON
// ============================================================================
//
// Serial (this file):
//   - All code runs on CPU
//   - One thread processes everything sequentially
//   - No #pragma directives needed
//   - Simple, easy to understand
//   - Slower for large problems
//
// Parallel (openmpandtest.c):
//   - Uses OpenMP directives (#pragma omp ...)
//   - Multiple threads/cores work simultaneously
//   - Can run on CPU (parallel for) or GPU (target)
//   - Faster for large problems
//   - More complex, requires careful data management
//
// ============================================================================

