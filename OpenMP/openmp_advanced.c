#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// ============================================================================
// OPENMP ADVANCED TOPICS
// ============================================================================
// This file demonstrates:
// 1. #pragma omp parallel with blocks
// 2. #pragma omp critical (mutual exclusion)
// 3. Variable scoping (shared, private, firstprivate, lastprivate)
// 4. Schedule clause (static, dynamic, guided, auto)
// 5. Parallel odd-even transport sort
// ============================================================================

// ============================================================================
// 1. PARALLEL BLOCK - Basic parallel region
// ============================================================================

void parallel_block_example() {
    printf("\n=== 1. Parallel Block Example ===\n");
    
    // Creates a team of threads
    // All threads execute the code block
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("Thread %d of %d says hello!\n", thread_id, num_threads);
    }
    // Implicit barrier here - all threads wait before continuing
    // this happenes automatically as it is implicit
}

// ============================================================================
// 2. CRITICAL SECTION - Mutual Exclusion
// ============================================================================

void critical_section_example() {
    printf("\n=== 2. Critical Section Example ===\n");
    
    int shared_counter = 0;
    
    // Without critical: data race (wrong results)
    printf("Without critical (may have data race):\n");
    shared_counter = 0;
    #pragma omp parallel
    {
        for (int i = 0; i < 1000; i++) {
            shared_counter++;  // DATA RACE!
        }
    }
    // this will give wrong results as multiple threads are modifying the same variable.
    printf("Counter (wrong): %d (should be %d)\n", shared_counter, 1000 * omp_get_max_threads());
    
    // With critical: only one thread at a time
    printf("\nWith critical (correct):\n");
    shared_counter = 0;
    #pragma omp parallel
    {
        for (int i = 0; i < 1000; i++) {
            #pragma omp critical//only one thread at a time can enter this block.
            {
                shared_counter++;  // Only one thread executes this at a time
            }
        }
    }
    printf("Counter (correct): %d\n", shared_counter);
}

// ============================================================================
// 3. VARIABLE SCOPING
// ============================================================================
//Does every thread look at the same variable, or does every thread get its own copy?
// shared(var) - all threads see the same variable
// private(var) - each thread has its own copy (uninitialized)
// firstprivate(var) - each thread has copy initialized to original value
// lastprivate(var) - each thread has private copy (last value copied back)
void variable_scoping_example() {
    printf("\n=== 3. Variable Scoping Example ===\n");
    
    int shared_var = 10;      // Shared by default
    int private_var = 20;      // Will be made private
    int firstprivate_var = 30; // Private, initialized from original
    int lastprivate_var = 40;  // Private, last value copied back
    
    printf("Before parallel: shared=%d, private=%d, firstprivate=%d, lastprivate=%d\n",
           shared_var, private_var, firstprivate_var, lastprivate_var);
    
    #pragma omp parallel shared(shared_var) private(private_var) \
        firstprivate(firstprivate_var) lastprivate(lastprivate_var)
    {
        int thread_id = omp_get_thread_num();
        
        // shared_var: all threads see the same variable
        shared_var += thread_id;
        
        // private_var: each thread has its own copy (uninitialized)
        private_var = thread_id * 10;
        
        // firstprivate_var: each thread has copy initialized to original value
        firstprivate_var += thread_id;
        
        // lastprivate_var: each thread has copy, last iteration's value copied back
        lastprivate_var = thread_id * 100;
        
        printf("Thread %d: shared=%d, private=%d, firstprivate=%d, lastprivate=%d\n",
               thread_id, shared_var, private_var, firstprivate_var, lastprivate_var);
    }
    
    printf("After parallel: shared=%d, private=%d (unchanged), firstprivate=%d (unchanged), lastprivate=%d\n",
           shared_var, private_var, firstprivate_var, lastprivate_var);
}

// ============================================================================
// 4. SCHEDULE CLAUSE - Loop iteration distribution
// ============================================================================

void schedule_clause_example(int *a, int n) {
    printf("\n=== 4. Schedule Clause Example ===\n");
    
    // Initialize array
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
    
    // STATIC: Divide iterations into chunks, assign to threads at compile time
    printf("\nStatic schedule (chunk size 4):\n");
    //chunk size is set to 4 for this example. 
    // Before loop starts:
    // Thread 0: Will process iterations 0-3, 16-19, 32-35, ...
    // Thread 1: Will process iterations 4-7, 20-23, 36-39, ...
    // Thread 2: Will process iterations 8-11, 24-27, 40-43, ...
    // Thread 3: Will process iterations 12-15, 28-31, 44-47, ...
    
    // This is FIXED - doesn't change during execution

    #pragma omp parallel for schedule(static, 4)
    for (int i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        printf("Thread %d processes iteration %d\n", tid, i);
    }
    
    // DYNAMIC: Threads grab chunks as they finish (load balancing)
    //chunk size is set to 2 for this example. 
// During loop execution:
// Thread 0: Grabs 0-3, finishes → grabs 16-19, finishes → grabs 32-35, ...
// Thread 1: Grabs 4-7, finishes → grabs 20-23, finishes → grabs 36-39, ...
// Thread 2: Grabs 8-11, finishes → grabs 24-27, finishes → grabs 40-43, ...
// Thread 3: Grabs 12-15, finishes → grabs 28-31, finishes → grabs 44-47, ...

// Assignment happens on-the-fly as threads finish
    printf("\nDynamic schedule (chunk size 2):\n");
    #pragma omp parallel for schedule(dynamic, 2)
    for (int i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        printf("Thread %d processes iteration %d\n", tid, i);
    }
    
    // GUIDED: Chunk size decreases as work progresses
    //chunk_size = remaining_iterations / number_of_threads
    printf("\nGuided schedule (chunk size starts large, decreases):\n");
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        printf("Thread %d processes iteration %d\n", tid, i);
    }
    
    // AUTO: Let OpenMP decide
    // OpenMP decides the best schedule based on the problem size and the number of threads.
    // This is sometimes but not always the default schedule if no schedule clause is specified.
    //depending on implimentation and hardware.
    printf("\nAuto schedule (OpenMP decides):\n");
    #pragma omp parallel for schedule(auto)
    for (int i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        printf("Thread %d processes iteration %d\n", tid, i);
    }
}

// ============================================================================
// 5. ODD-EVEN TRANSPORT SORT - Parallel Implementation
// ============================================================================
//sorting algorithm allows elements to move in both directuins
//even phase: compare and swap (0,1), (2,3), (4,5), psition 0-1, 2-3
//odd phase: compare and swap (1,2), (3,4), (5,6), position 1-2, 3-4
void odd_even_transport_sort_serial(int *arr, int n) {
    int sorted = 0;
    while (!sorted) {
        sorted = 1;
        
        // Odd phase: compare and swap (1,2), (3,4), (5,6), ...
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                sorted = 0;
            }
        }
        
        // Even phase: compare and swap (0,1), (2,3), (4,5), ...
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                sorted = 0;
            }
        }
    }
}

void odd_even_transport_sort_parallel(int *arr, int n) {
    int sorted = 0;
    int phase = 0;  // 0 = even phase, 1 = odd phase
    
    while (!sorted) {
        sorted = 1;
        
        // Parallelize the comparison and swap operations
        // Use critical section to update sorted flag safely
        #pragma omp parallel
        {
            int local_sorted = 1;
            
            if (phase == 0) {
                // Even phase: (0,1), (2,3), (4,5), ...
                #pragma omp for schedule(static)// predifined chunks
                for (int i = 0; i < n - 1; i += 2) {
                    if (arr[i] > arr[i + 1]) {
                        int temp = arr[i];
                        arr[i] = arr[i + 1];
                        arr[i + 1] = temp;
                        local_sorted = 0;
                    }
                }
            } else {
                // Odd phase: (1,2), (3,4), (5,6), ...
                #pragma omp for schedule(static)
                for (int i = 1; i < n - 1; i += 2) {
                    if (arr[i] > arr[i + 1]) {
                        int temp = arr[i];
                        arr[i] = arr[i + 1];
                        arr[i + 1] = temp;
                        local_sorted = 0;
                    }
                }
            }
            
            // Update global sorted flag (critical section)
            if (!local_sorted) {
                #pragma omp critical //only one thread at a time can enter this block.
                {
                    sorted = 0;
                }
            }
        }
        
        phase = 1 - phase;  // Toggle between even and odd
    }
}

// ============================================================================
// MAIN FUNCTION - Demonstrates all features
// ============================================================================

int main() {
    printf("OpenMP Advanced Topics Demonstration\n");
    printf("=====================================\n");
    
    // Set number of threads
    omp_set_num_threads(4);
    printf("Using %d threads\n", omp_get_max_threads());
    
    // 1. Parallel block
    parallel_block_example();
    
    // 2. Critical section
    critical_section_example();
    
    // 3. Variable scoping
    variable_scoping_example();
    
    // 4. Schedule clause
    const int n = 16;
    int *schedule_array = (int*)malloc(n * sizeof(int));
    schedule_clause_example(schedule_array, n);
    free(schedule_array);
    
    // 5. Odd-even transport sort
    printf("\n=== 5. Odd-Even Transport Sort ===\n");
    const int sort_size = 20;
    int *arr_serial = (int*)malloc(sort_size * sizeof(int));
    int *arr_parallel = (int*)malloc(sort_size * sizeof(int));
    
    // Initialize with random values
    srand(time(NULL));
    printf("\nOriginal array:\n");
    for (int i = 0; i < sort_size; i++) {
        arr_serial[i] = arr_parallel[i] = rand() % 100;
        printf("%d ", arr_serial[i]);
    }
    printf("\n");
    
    // Serial sort
    clock_t start = clock();
    odd_even_transport_sort_serial(arr_serial, sort_size);
    clock_t serial_time = clock() - start;
    
    printf("\nSerial sorted array:\n");
    for (int i = 0; i < sort_size; i++) {
        printf("%d ", arr_serial[i]);
    }
    printf("\nSerial time: %ld clock ticks\n", serial_time);
    
    // Parallel sort
    start = clock();
    odd_even_transport_sort_parallel(arr_parallel, sort_size);
    clock_t parallel_time = clock() - start;
    
    printf("\nParallel sorted array:\n");
    for (int i = 0; i < sort_size; i++) {
        printf("%d ", arr_parallel[i]);
    }
    printf("\nParallel time: %ld clock ticks\n", parallel_time);
    
    // Verify correctness
    int correct = 1;
    for (int i = 0; i < sort_size; i++) {
        if (arr_serial[i] != arr_parallel[i]) {
            correct = 0;
            break;
        }
    }
    printf("\nSort correctness: %s\n", correct ? "PASS" : "FAIL");
    
    free(arr_serial);
    free(arr_parallel);
    
    return 0;
}

// ============================================================================
// REFERENCE GUIDE
// ============================================================================
//
// PARALLEL REGION:
//   #pragma omp parallel
//   {
//       // Code executed by all threads
//   }
//
// CRITICAL SECTION:
//   #pragma omp critical
//   {
//       // Only one thread executes at a time
//   }
//
// VARIABLE SCOPING:
//   shared(var)      - All threads share same variable
//   private(var)     - Each thread has private copy (uninitialized)
//   firstprivate(var) - Each thread has private copy (initialized from original)
//   lastprivate(var)  - Each thread has private copy (last value copied back)
//   default(shared)   - Default: variables are shared
//   default(none)     - Must explicitly specify all variable scoping
//
// SCHEDULE CLAUSES:
//   schedule(static)           - Divide into equal chunks at compile time
//   schedule(static, chunk)   - Each chunk has 'chunk' iterations
//   schedule(dynamic)         - Threads grab chunks as they finish
//   schedule(dynamic, chunk)  - Dynamic with specified chunk size
//   schedule(guided)         - Chunk size decreases as work progresses
//   schedule(guided, chunk)  - Guided with minimum chunk size
//   schedule(auto)            - Let OpenMP decide
//   schedule(runtime)         - Set via OMP_SCHEDULE environment variable
//
// ODD-EVEN TRANSPORT SORT:
//   - Alternates between even and odd phases
//   - Even phase: compare pairs (0,1), (2,3), (4,5), ...
//   - Odd phase: compare pairs (1,2), (3,4), (5,6), ...
//   - Parallelizable because comparisons in each phase are independent
//   - Uses critical section to safely update global sorted flag
//
// ============================================================================

