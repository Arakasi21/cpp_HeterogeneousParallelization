/*
 * Лабораторная работа: Редукция суммы элементов массива
 * Задача 2: Оптимизация параллельного редукционного алгоритма
 * 
 * Сравниваем два подхода:
 * a) Только глобальная память
 * b) Глобальная + разделяемая память (shared memory)
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace std::chrono;

// все операции будут идти через глобальную память
// глобальная память медленная
// поэтому этот вариант будет работать медленно

__global__ void reductionGlobalOnly(float* d_input, float* d_output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // указатель на начало данных для этого блока
    // каждый блок обрабатывает свой кусок массива
    int blockStart = blockIdx.x * blockDim.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (blockStart + tid + stride) < N) {
            d_input[blockStart + tid] += d_input[blockStart + tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_output[blockIdx.x] = d_input[blockStart];
    }
}

// shared memory - это быстрая память внутри каждого SM

__global__ void reductionSharedMemory(float* d_input, float* d_output, int N) {
    // объявляем разделяемую память
    // __shared__ означает что эта память общая для всех потоков в блоке
    // extern - размер задается при запуске kernel
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        sdata[tid] = d_input[idx];
    } else {
        sdata[tid] = 0.0f;  // padding нулями для потоков за границей
    }
    __syncthreads();
    // это быстро потому что shared memory очень быстрая
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}
// warp - это группа из 32 потоков которые выполняются вместе
// внутри warp не нужна синхронизация __syncthreads()

__global__ void reductionOptimized(float* d_input, float* d_output, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    float mySum = 0.0f;
    if (idx < N) {
        mySum = d_input[idx];
    }
    if (idx + blockDim.x < N) {
        mySum += d_input[idx + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();
    
    // редукция в shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // внутри одного warp все потоки выполняются синхронно (SIMT)
    // поэтому __syncthreads() не нужен
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}
float finalReductionCPU(float* h_partialSums, int numBlocks) {
    float sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partialSums[i];
    }
    return sum;
}
float cpuReduction(float* h_input, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += h_input[i];
    }
    return sum;
}
struct TestResult {
    double time;
    float sum;
    bool correct;
};

TestResult testGlobalMemory(float* d_input, float* d_temp, int N, int blockSize, float expectedSum) {
    TestResult result;
    
    int numBlocks = (N + blockSize - 1) / blockSize;
    float* h_partial = new float[numBlocks];
    
    // делаем копию входных данных
    float* d_inputCopy;
    cudaMalloc((void**)&d_inputCopy, N * sizeof(float));
    cudaMemcpy(d_inputCopy, d_input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    
    auto start = high_resolution_clock::now();
    
    // первый проход редукции
    reductionGlobalOnly<<<numBlocks, blockSize>>>(d_inputCopy, d_temp, N);
    cudaDeviceSynchronize();
    
    // копируем частичные суммы на host
    cudaMemcpy(h_partial, d_temp, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // финальная редукция на CPU
    result.sum = finalReductionCPU(h_partial, numBlocks);
    
    auto end = high_resolution_clock::now();
    duration<double, milli> elapsed = end - start;
    
    result.time = elapsed.count();
    result.correct = fabs(result.sum - expectedSum) < expectedSum * 0.001f; 
    
    cudaFree(d_inputCopy);
    delete[] h_partial;
    
    return result;
}

// тестирование редукции с shared memory
TestResult testSharedMemory(float* d_input, float* d_temp, int N, int blockSize, float expectedSum) {
    TestResult result;
    
    int numBlocks = (N + blockSize - 1) / blockSize;
    float* h_partial = new float[numBlocks];
    size_t sharedMemSize = blockSize * sizeof(float);
    
    auto start = high_resolution_clock::now();
    
    // запуск с указанием размера shared memory
    // третий параметр <<< >>> - размер динамической shared memory
    reductionSharedMemory<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_temp, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_partial, d_temp, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    result.sum = finalReductionCPU(h_partial, numBlocks);
    
    auto end = high_resolution_clock::now();
    duration<double, milli> elapsed = end - start;
    
    result.time = elapsed.count();
    result.correct = fabs(result.sum - expectedSum) < expectedSum * 0.001f;
    
    delete[] h_partial;
    
    return result;
}

TestResult testOptimized(float* d_input, float* d_temp, int N, int blockSize, float expectedSum) {
    TestResult result;
    
    // в оптимизированной версии каждый поток обрабатывает 2 элемента
    int numBlocks = (N + blockSize * 2 - 1) / (blockSize * 2);
    float* h_partial = new float[numBlocks];
    
    size_t sharedMemSize = blockSize * sizeof(float);
    
    auto start = high_resolution_clock::now();
    
    reductionOptimized<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_temp, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_partial, d_temp, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    result.sum = finalReductionCPU(h_partial, numBlocks);
    
    auto end = high_resolution_clock::now();
    duration<double, milli> elapsed = end - start;
    
    result.time = elapsed.count();
    result.correct = fabs(result.sum - expectedSum) < expectedSum * 0.001f;
    
    delete[] h_partial;
    
    return result;
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    cout << "========================================" << endl;
    cout << " sum reduction" << endl;
    cout << "========================================" << endl << endl;
    
    cout << "array size: " << N << " elements" << endl;
    cout << "memory: " << size / 1024 / 1024 << " MB" << endl << endl;
    
    // выделяем память на host
    float* h_input = new float[N];
    
    // инициализируем массив
    // используем маленькие числа чтобы избежать overflow при суммировании
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;  // от 0.0 до 1.0
    }
    
    // вычисляем эталонную сумму на CPU
    auto cpuStart = high_resolution_clock::now();
    float cpuSum = cpuReduction(h_input, N);
    auto cpuEnd = high_resolution_clock::now();
    duration<double, milli> cpuTime = cpuEnd - cpuStart;
    
    cout << "CPU sum: " << fixed << setprecision(2) << cpuSum << endl;
    cout << "CPU time: " << fixed << setprecision(2) << cpuTime.count() << " ms" << endl << endl;
    
    // выделяем память на GPU
    float *d_input, *d_temp;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_temp, size); 
    
    // копируем данные на GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    
    cout << "1. Global memory only:" << endl;
    TestResult globalResult = testGlobalMemory(d_input, d_temp, N, blockSize, cpuSum);
    cout << "   sum: " << fixed << setprecision(2) << globalResult.sum << endl;
    cout << "   time: " << fixed << setprecision(2) << globalResult.time << " ms" << endl;
    cout << "   correctness: " << (globalResult.correct ? "[OK]" : "[FAIL]") << endl << endl;
    
    // ТЕСТ 2: shared memory
    cout << "2. Shared memory:" << endl;
    TestResult sharedResult = testSharedMemory(d_input, d_temp, N, blockSize, cpuSum);
    cout << "   sum: " << fixed << setprecision(2) << sharedResult.sum << endl;
    cout << "   time: " << fixed << setprecision(2) << sharedResult.time << " ms" << endl;
    cout << "   correctness: " << (sharedResult.correct ? "[OK]" : "[FAIL]") << endl << endl;
    
    // ТЕСТ 3: оптимизированная версия
    cout << "3. Optimized :" << endl;
    TestResult optResult = testOptimized(d_input, d_temp, N, blockSize, cpuSum);
    cout << "   sum: " << fixed << setprecision(2) << optResult.sum << endl;
    cout << "   time: " << fixed << setprecision(2) << optResult.time << " ms" << endl;
    cout << "   correctness: " << (optResult.correct ? "[OK]" : "[FAIL]") << endl << endl;
    
    // сравнение результатов
    cout << "========================================" << endl;
    
    cout << "| Method               | Time     | Speedup  |" << endl;
    cout << "| CPU (sequential)     | " << setw(6) << fixed << setprecision(2) 
         << cpuTime.count() << " ms | 1.00x    |" << endl;
    cout << "| GPU Global Only      | " << setw(6) << fixed << setprecision(2) 
         << globalResult.time << " ms | " << setw(5) << setprecision(2) 
         << cpuTime.count() / globalResult.time << "x   |" << endl;
    cout << "| GPU Shared Memory    | " << setw(6) << fixed << setprecision(2) 
         << sharedResult.time << " ms | " << setw(5) << setprecision(2) 
         << cpuTime.count() / sharedResult.time << "x   |" << endl;
    cout << "| GPU Optimized        | " << setw(6) << fixed << setprecision(2) 
         << optResult.time << " ms | " << setw(5) << setprecision(2) 
         << cpuTime.count() / optResult.time << "x   |" << endl;

    cout << "Speedup shared vs global: " << fixed << setprecision(2) 
         << globalResult.time / sharedResult.time << "x" << endl;
    cout << "Speedup optimized vs global: " << fixed << setprecision(2) 
         << globalResult.time / optResult.time << "x" << endl;
         
    cudaFree(d_input);
    cudaFree(d_temp);
    delete[] h_input;
    
    return 0;
}
