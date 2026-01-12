#include <iostream> // cin cout
#include <cuda_runtime.h> // для CUDA функций
#include <chrono> // high resolution clock для замера времени
#include <iomanip> // setprecision, fixed

using namespace std;
using namespace std::chrono;

// COALESCED ACCESS
// Kernel с коалесцированным доступом к памяти. Коалесцирование означает, что доступ к памяти происходит последовательно, что позволяет GPU эффективно группировать обращения к памяти
// Соседние потоки читают соседние адреса памяти
// Thread 0 читает element[0], Thread 1 читает element[1], и т.д.
// GPU может объединить эти обращения в одну транзакцию памяти
__global__ void coalescedAccess(float* d_in, float* d_out, int N) {
    // вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // например 
        // Thread 0 → d_in[0]
        // Thread 1 → d_in[1]
        // Thread 2 → d_in[2]
        // Все потоки в warp (32 потока) читают последовательный блок памяти
        // GPU объединяет это в одну транзакцию памяти
        float value = d_in[idx];
        d_out[idx] = value * 2.0f + 1.0f;
    }
}

// непоследовательный доступ 
// Kernel с некоалесцированным доступом к памяти
// Потоки читают память с stride (шагом), разбросаны по памяти
// Это заставляет GPU делать много отдельных транзакций памяти
__global__ void nonCoalescedAccess(float* d_in, float* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Thread 0 → d_in[0]
        // Thread 1 → d_in[32]
        // Thread 2 → d_in[64]
        // Потоки в warp читают данные разбросанные по памяти
        // GPU не может объединить это → много отдельных транзакций памяти
        
        int stride = 32;
        int strided_idx = (idx / stride) + (idx % stride) * (N / stride);
        
        if (strided_idx < N) {
            float value = d_in[strided_idx];
            d_out[strided_idx] = value * 2.0f + 1.0f;
        }
    }
}

int main() {
    int N = 10000000; 
    size_t size = N * sizeof(float);
    
    cout << " coalescing test " << endl;
    cout << N << " elements" << endl;
    cout << "memory per array: " << size / 1024 / 1024 << " MB" << endl;
    cout << "total memory: " << 2 * size / 1024 / 1024 << " MB" << endl << endl;
    
    float* h_in = new float[N];
    float* h_out_coalesced = new float[N];
    float* h_out_strided = new float[N];

    srand(time(0));
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 100) / 10.0f;
    }
    
    cout << "first 5 input elements: ";
    for (int i = 0; i < 5; i++) cout << h_in[i] << " ";
    cout << endl << endl;

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int blocksPerGrid = (N + blockSize - 1) / blockSize;
    
    cout << "block size: " << blockSize << " threads" << endl;
    cout << "blocks: " << blocksPerGrid << endl;
    cout << "total threads: " << blocksPerGrid * blockSize << endl << endl;
    
    cout << "coalesced access" << endl;
    
    // warm-up run
    coalescedAccess<<<blocksPerGrid, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    const int NUM_RUNS = 5;
    double coalesced_time = 0.0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        auto start = high_resolution_clock::now();
        
        coalescedAccess<<<blocksPerGrid, blockSize>>>(d_in, d_out, N);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
        }
        
        cudaDeviceSynchronize();
        
        auto end = high_resolution_clock::now();
        duration<double> elapsed = end - start;
        coalesced_time += elapsed.count();
    }
    coalesced_time /= NUM_RUNS;

    cudaMemcpy(h_out_coalesced, d_out, size, cudaMemcpyDeviceToHost);
    
    cout << "avg time (" << NUM_RUNS << " runs): " << fixed << setprecision(6) 
         << coalesced_time << " sec" << endl;
    cout << "result: ";
    for (int i = 0; i < 5; i++) {
        cout << h_out_coalesced[i] << " ";
    }
    cout << endl << endl;

    cout << "non-coalesced access (strided)" << endl;
    
    // прогрев
    nonCoalescedAccess<<<blocksPerGrid, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    double strided_time = 0.0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        auto start = high_resolution_clock::now();
        
        nonCoalescedAccess<<<blocksPerGrid, blockSize>>>(d_in, d_out, N);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
        }
        
        cudaDeviceSynchronize();
        
        auto end = high_resolution_clock::now();
        duration<double> elapsed = end - start;
        strided_time += elapsed.count();
    }
    strided_time /= NUM_RUNS;
    
    cudaMemcpy(h_out_strided, d_out, size, cudaMemcpyDeviceToHost);
    
    cout << "avg time (" << NUM_RUNS << " runs): " << fixed << setprecision(6) 
         << strided_time << " sec" << endl;
    cout << "result: ";
    for (int i = 0; i < 5; i++) {
        cout << h_out_strided[i] << " ";
    }
    cout << endl << endl;
    
    cout << "final results" << endl << endl;
    
    cout << "+-------------------+------------+----------------+------------------+" << endl;
    cout << "| Access Pattern    | Time (sec) | Relative Speed | Slowdown Factor  |" << endl;
    cout << "+-------------------+------------+----------------+------------------+" << endl;
    
    double best_time = coalesced_time;
    
    cout << "| Coalesced         | " << fixed << setprecision(6) << setw(10) << coalesced_time
         << " | " << setw(14) << "1.00x" << " | " << setw(16) << "Baseline" << " |" << endl; // baseline для сравнения
    
    cout << "| Strided (32)      | " << fixed << setprecision(6) << setw(10) << strided_time
         << " | " << fixed << setprecision(2) << setw(14) << (best_time / strided_time) << "x"
         << " | " << fixed << setprecision(2) << setw(16) << (strided_time / best_time) << "x slower |" << endl;
    cout << "+-------------------+------------+----------------+------------------+" << endl << endl;
    double strided_slowdown_pct = ((strided_time / coalesced_time) - 1.0) * 100.0;
    
    cout << "  Strided access is " << fixed << setprecision(1) << strided_slowdown_pct 
         << "% slower than coalesced" << endl;
    
    cout << "Performance:" << endl;
    cout << "-----------------------------------------------------------" << endl;
    
    int bar1 = 50; 
    int bar2 = (int)((coalesced_time / strided_time) * 50);
    
    cout << "Coalesced  | ";
    for (int i = 0; i < bar1; i++) cout << "#";
    cout << " 1.00x" << endl;
    
    cout << "Strided    | ";
    for (int i = 0; i < bar2; i++) cout << "#";
    cout << " " << fixed << setprecision(2) << (coalesced_time / strided_time) << "x" << endl;
    
    cout << "-----------------------------------------------------------" << endl << endl;
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out_coalesced;
    delete[] h_out_strided;
    
    return 0;
}