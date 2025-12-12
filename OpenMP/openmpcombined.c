#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// ============================================================================
// OPENMP DIRECTIVES OVERVIEW
// ============================================================================

// CPU Parallelism (runs on CPU cores)
// #pragma omp parallel for - parallelize loop across CPU threads

// GPU Offloading (runs on GPU)
// #pragma omp target - offload code block to GPU
// #pragma omp target teams - create thread teams on GPU
// #pragma omp target teams distribute - split loop iterations across teams
// #pragma omp target teams distribute parallel for - full GPU parallelization

// ============================================================================
// 1. CPU PARALLELIZATION
// ============================================================================

void cpu_parallel_example(int *a, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = i * 2;  // Runs on CPU cores in parallel
    }
}
void serial_example(int *a, int n) {
    // Serial loop - runs sequentially on one CPU thread
    for (int i = 0; i < n; i++) {
        a[i] = i * 2;
    }
}

// ============================================================================
// 2. GPU OFFLOADING - BASIC
// ============================================================================

void gpu_basic_example() {
    #pragma omp target
    {
        // Code block runs on GPU
        printf("Hello from GPU!\n");
    }
}


void basic_example() {
    // Code runs on CPU sequentially
    printf("Hello from CPU!\n");
}
// ============================================================================
// 3. GPU OFFLOADING - VECTOR ADDITION (Complete Example)
// ============================================================================

void vec_add(float *a, float *b, float *c, int n) {
    // map(to: ...) - copy data TO GPU
    // map(from: ...) - copy data FROM GPU back to CPU
    // map(tofrom: ...) - copy both directions
    
    #pragma omp target teams distribute parallel for 
        map(to: a[0:n], b[0:n]) 
        map(from: c[0:n])
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void vec_add(float *a, float *b, float *c, int n) {
    // Serial loop - processes elements one at a time
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// 4. REDUCTION - CORRECT vs INCORRECT
// ============================================================================

//put for race conditions
// ❌ WRONG: Data race - multiple threads write to same variable
void bad_reduction(double *a, int n) {
    double sum = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        sum += a[i];  // DATA RACE! Multiple threads modifying sum
    }
}

// ✅ CORRECT: Use reduction clause
void good_reduction_cpu(double *a, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i];  // Each thread has private copy, combined at end
    }
}


void serial_reduction(double *a, int n) {
    double sum = 0.0;
    // Serial loop - no parallelization, so no data race
    for (int i = 0; i < n; i++) {
        sum += a[i];  // Safe in serial - only one thread
    }
    printf("Sum: %.2f\n", sum);
}

// ✅ CORRECT: GPU reduction with mapping
void good_reduction_gpu(double *a, int n) {
    double sum = 0.0;
    #pragma omp target teams distribute parallel for \
        reduction(+:sum) \
        map(to: a[0:n]) map(tofrom: sum)
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
}
// reduction clauses
reduction(+:sum)    // Sum: sum = sum1 + sum2 + sum3 + ...
reduction(*:prod)   // Product: prod = prod1 * prod2 * prod3 * ...
reduction(max:max_val)  // Maximum: max_val = max(max1, max2, max3, ...)
reduction(min:min_val)  // Minimum: min_val = min(min1, min2, min3, ...)

// ============================================================================
// 5. COLLAPSE - Nested Loops
// ============================================================================

void nested_loops_example(float *A, float *B, int n, int m) {
    // collapse(2) - parallelize both nested loops as one
    // Without collapse: only outer loop parallelized (less parallelism)
    // With collapse: both loops combined into one parallel loop (more parallelism)
    
    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: A[0:n*m]) map(from: B[0:n*m])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int idx = i * m + j;
            B[idx] = A[idx] * 2.0f;
        }
    }
}


void nested_loops_example_serial(float *A, float *B, int n, int m) {
    // Serial nested loops - processes all iterations sequentially
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int idx = i * m + j;
            B[idx] = A[idx] * 2.0f;
        }
    }
}

// ============================================================================
// 6. COMPLETE EXAMPLE - Dot Product (All Directives Together)
// ============================================================================

double dot_product(double *a, double *b, int n) {
    double sum = 0.0;
    // target - offload to GPU
    // teams - create thread teams
    // distribute - distribute iterations across teams
    // parallel for - parallelize loop within teams
    // reduction(+:sum) - combine partial sums
    // map(to: ...) - copy input arrays to GPU
    // map(tofrom: sum) - copy sum to/from GPU
    #pragma omp target teams distribute parallel for \
        map(to: a[0:n], b[0:n]) \
        map(tofrom: sum) \
        reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
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
    // Example: Vector addition on GPU
    vec_add(a, b, c, n);
    // Print first few results
    printf("Vector addition results (first 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %.1f\n", i, c[i]);
    }
    
    // Example: Dot product on GPU
    double *da = (double*)malloc(n * sizeof(double));// Allocate da
    double *db = (double*)malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        da[i] = i;// Initialize da with values 0, 1, 2, ..
        db[i] = i;// Initialize db with values 0, 1, 2, ..
    }
    double result = dot_product(da, db, n); // Use these arrays for dot product
    printf("\nDot product result: %.2f\n", result);
    // Cleanup
    free(a);
    free(b);
    free(c);
    free(da);
    free(db);
    
    return 0;
}

// ============================================================================
// DIRECTIVE REFERENCE
// ============================================================================
// 
// CPU Directives:
//   #pragma omp parallel for          - Parallelize loop on CPU
//   #pragma omp parallel for reduction(op:var) - Parallel with reduction
//
// GPU Directives:
//   #pragma omp target                - Offload code block to GPU
//   #pragma omp target teams           - Create thread teams on GPU
//   #pragma omp target teams distribute - Distribute iterations to teams
//   #pragma omp target teams distribute parallel for - Full GPU parallelization
//
// Clauses:
//   map(to: arr[0:n])      - Copy array TO GPU (input)
//   map(from: arr[0:n])    - Copy array FROM GPU (output)
//   map(tofrom: arr[0:n])  - Copy array both directions (input/output)
//   reduction(op:var)      - Combine results from all threads
//   collapse(k)            - Collapse k nested loops into one
//
// ============================================================================
