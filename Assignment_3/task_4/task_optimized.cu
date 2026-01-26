#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// Global Memory
__global__ void multiplyGlobalMemory(int* d_in, int* d_out, int N, int multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        d_out[idx] = d_in[idx] * multiplier;
    }
}

// Shared Memory
__global__ void multiplySharedMemory(int* d_in, int* d_out, int N, int multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ int s_data[];
    
    if (idx < N) {
        s_data[threadIdx.x] = d_in[idx];
    }
    
    __syncthreads();
    
    if (idx < N) {
        d_out[idx] = s_data[threadIdx.x] * multiplier;
    }
}

// BENCHMARK STRUCTURE

struct BenchmarkResult {
    int blockSize;          
    int blocksPerGrid;       
    double globalMemTime;    
    double sharedMemTime;    
    double speedup;        
};

// BENCHMARK FUNCTION

BenchmarkResult testConfiguration(int* d_in, int* d_out, int N, int multiplier, 
                                  int blockSize, int numRuns = 5) {
    BenchmarkResult result;
    result.blockSize = blockSize;
    result.blocksPerGrid = (N + blockSize - 1) / blockSize;
    
    size_t sharedMemSize = blockSize * sizeof(int);

    multiplyGlobalMemory<<<result.blocksPerGrid, blockSize>>>(d_in, d_out, N, multiplier);
    cudaDeviceSynchronize();
    
    multiplySharedMemory<<<result.blocksPerGrid, blockSize, sharedMemSize>>>(d_in, d_out, N, multiplier);
    cudaDeviceSynchronize();

    double totalTimeGlobal = 0.0;
    for (int run = 0; run < numRuns; run++) {
        cudaMemset(d_out, 0, N * sizeof(int));
        
        auto start = high_resolution_clock::now();
        multiplyGlobalMemory<<<result.blocksPerGrid, blockSize>>>(d_in, d_out, N, multiplier);
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        
        duration<double> elapsed = end - start;
        totalTimeGlobal += elapsed.count();
    }
    result.globalMemTime = totalTimeGlobal / numRuns;
    
    double totalTimeShared = 0.0;
    for (int run = 0; run < numRuns; run++) {
        cudaMemset(d_out, 0, N * sizeof(int));
        
        auto start = high_resolution_clock::now();
        multiplySharedMemory<<<result.blocksPerGrid, blockSize, sharedMemSize>>>(d_in, d_out, N, multiplier);
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        
        duration<double> elapsed = end - start;
        totalTimeShared += elapsed.count();
    }
    result.sharedMemTime = totalTimeShared / numRuns;
    
    result.speedup = result.globalMemTime / result.sharedMemTime;
    
    return result;
}

// UTILITY FUNCTIONS

void printTableHeader() {
    cout << "+------------+--------+--------------+--------------+----------+---------+" << endl;
    cout << "| Block Size | Blocks | Global (sec) | Shared (sec) | Best Ver | Speedup |" << endl;
    cout << "+------------+--------+--------------+--------------+----------+---------+" << endl;
}

void printTableRow(const BenchmarkResult& result, bool isBest) {
    cout << "| " << setw(10) << result.blockSize 
         << " | " << setw(6) << result.blocksPerGrid
         << " | " << fixed << setprecision(6) << setw(12) << result.globalMemTime
         << " | " << fixed << setprecision(6) << setw(12) << result.sharedMemTime
         << " | " << setw(8) << (result.speedup > 1.0 ? "Shared" : "Global")
         << " | " << fixed << setprecision(2) << setw(7) << abs(result.speedup) << "x";
    
    if (isBest) {
        cout << " | <-- OPTIMAL";
    }
    cout << endl;
}

void printSeparator() {
    cout << "========================================" << endl;
}

// MAIN 

int main() {
    printSeparator();
    cout << "   CUDA BLOCK SIZE OPTIMIZATION TEST" << endl;
    printSeparator();
    cout << endl;
    
    int N = 1000000000; 
    int multiplier = 5;
    size_t size = N * sizeof(int);
    
    cout << "Config:" << endl;
    cout << "  Array size: " << N << " elements" << endl;
    cout << "  Multiplier: " << multiplier << endl;
    cout << "  Memory per array: " << size / 1024 / 1024 << " MB" << endl << endl;
    
    int* h_in = new int[N];
    int* h_out = new int[N];
    
    srand(time(0));
    for (int i = 0; i < N; i++) {
        h_in[i] = rand() % 10 + 1;
    }
    
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    cout << "GPU Information:" << endl;
    cout << "  Device: " << deviceProp.name << endl;
    cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
    cout << "  Warp size: " << deviceProp.warpSize << endl;
    cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << endl << endl;
    
    // ========== ТЕСТИРУЕМЫЕ КОНФИГУРАЦИИ ==========
    vector<int> blockSizes = {32, 64, 128, 256, 512, 768, 1024};
    vector<BenchmarkResult> results;
    
    cout << "Testing" << endl;
    cout << "(Each configuration is tested 5 times, the average is displayed)" << endl << endl;
    
    // ========== ЗАПУСК ТЕСТОВ ==========
    int totalTests = blockSizes.size();
    for (int i = 0; i < totalTests; i++) {
        int blockSize = blockSizes[i];
        cout << "[" << (i + 1) << "/" << totalTests << "] test of block sizes" 
             << blockSize << "..." << flush;
        
        BenchmarkResult result = testConfiguration(d_in, d_out, N, multiplier, blockSize);
        results.push_back(result);
      }
    
    cout << endl;
    
    auto bestGlobal = min_element(results.begin(), results.end(), 
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.globalMemTime < b.globalMemTime;
        });
    
    auto bestShared = min_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.sharedMemTime < b.sharedMemTime;
        });
    
    auto bestOverall = min_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return min(a.globalMemTime, a.sharedMemTime) < min(b.globalMemTime, b.sharedMemTime);
        });
    
    cout << "FULL TEST RESULTS:" << endl;
    printTableHeader();
    
    for (const auto& result : results) {
        bool isBest = (&result == &(*bestOverall));
        printTableRow(result, isBest);
    }
    
    cout << endl;
    
    cout << "Global Memory:" << endl;
    cout << "  Best block size: " << bestGlobal->blockSize << " threads" << endl;
    cout << "  Best time: " << fixed << setprecision(6) << bestGlobal->globalMemTime << " sec" << endl;
    cout << "  Blocks per grid: " << bestGlobal->blocksPerGrid << endl << endl;
    
    cout << "Shared Memory:" << endl;
    cout << "  Best block size: " << bestShared->blockSize << " threads" << endl;
    cout << "  Best time: " << fixed << setprecision(6) << bestShared->sharedMemTime << " sec" << endl;
    cout << "  Blocks per grid: " << bestShared->blocksPerGrid << endl << endl;
    
    cout << "Optimal configuration:" << endl;
    cout << "  Block size: " << bestOverall->blockSize << " threads" << endl;
    cout << "  Memory type: " << (bestOverall->globalMemTime < bestOverall->sharedMemTime ? "Global" : "Shared") << endl;
    cout << "  Time: " << fixed << setprecision(6) 
         << min(bestOverall->globalMemTime, bestOverall->sharedMemTime) << " sec" << endl << endl;
    
    auto original = find_if(results.begin(), results.end(),
        [](const BenchmarkResult& r) { return r.blockSize == 256; });
    
    if (original != results.end()) {
        printSeparator();
        cout << "  COMPARISON WITH ORIG" << endl;
        printSeparator();
        cout << endl;
        
        double originalBestTime = min(original->globalMemTime, original->sharedMemTime);
        double optimizedBestTime = min(bestOverall->globalMemTime, bestOverall->sharedMemTime);
        double improvement = ((originalBestTime / optimizedBestTime) - 1.0) * 100.0;
        
        cout << "Original configuration (256 threads):" << endl;
        cout << "  Global memory: " << fixed << setprecision(6) << original->globalMemTime << " sec" << endl;
        cout << "  Shared memory: " << fixed << setprecision(6) << original->sharedMemTime << " sec" << endl;
        cout << "  Best: " << fixed << setprecision(6) << originalBestTime << " sec" << endl << endl;
        
        cout << "Optimized configuration (" << bestOverall->blockSize << " threads):" << endl;
        cout << "  Best time: " << fixed << setprecision(6) << optimizedBestTime << " sec" << endl;
        cout << "  Speedup: " << fixed << setprecision(2) << (originalBestTime / optimizedBestTime) << "x" << endl;
        
        if (improvement > 0) {
            cout << "  Performance improvement: +" << fixed << setprecision(1) << improvement << "%" << endl;
        } else {
            cout << "  Performance change: " << fixed << setprecision(1) << improvement << "%" << endl;
        }
        cout << endl;
    }
    
    
    cout << "CORRECTNESS CHECK:" << endl;
    multiplyGlobalMemory<<<bestOverall->blocksPerGrid, bestOverall->blockSize>>>(d_in, d_out, N, multiplier);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    bool correct = true;
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i] * multiplier) {
            correct = false;
            errors++;
            if (errors <= 3) {
                cout << "Error at index " << i << ": expected " << (h_in[i] * multiplier) 
                     << ", got " << h_out[i] << endl;
            }
        }
    }
    if (correct) {
        cout << "  Result is correct!" << endl;
    } else {
        cout << "  Result is incorrect! Total errors: " << errors << endl;
    }
    cout << endl << endl;
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    
    return 0;
}
