/*
 * Лабораторная работа: Гибридные и распределённые параллельные вычисления
 * Задание 2: Префиксная сумма (Scan) с разделяемой памятью
 * 
 * Префиксная сумма: output[i] = input[0] + input[1] + ... + input[i]
 * Пример: input = [3, 1, 7, 0, 4, 1, 6, 3]
 *        output = [3, 4, 11, 11, 15, 16, 22, 25]
 * 
 * Размер массива: 1,000,000 элементов
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 256

// Используем алгоритм Hillis-Steele для параллельного scan

__global__ void prefixScanShared(float* d_input, float* d_output, float* d_blockSums, int N) {
    // размер = 2 * BLOCK_SIZE для двойной буферизации
    __shared__ float temp[2 * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // каждый поток загружает один элемент в shared memory
    int pout = 0;  
    int pin = 1; 
    
    // загружаем данные в shared memory
    if (idx < N) {
        temp[tid] = d_input[idx];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();
    
    // алгоритм Hillis-Steele для inclusive scan
    // на каждом шаге offset удваивается: 1, 2, 4, 8, ...
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        // меняем местами входной и выходной буферы
        pout = 1 - pout;
        pin = 1 - pout;
        
        if (tid >= offset) {
            // складываем текущий элемент с элементом на расстоянии offset
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid] 
                                           + temp[pin * blockDim.x + tid - offset];
        } else {
            // просто копируем элемент
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid];
        }
        __syncthreads();
    }
    
    // записываем результат в глобальную память
    if (idx < N) {
        d_output[idx] = temp[pout * blockDim.x + tid];
    }
    
    // последний поток блока сохраняет сумму блока
    // это нужно для объединения результатов между блоками
    if (tid == blockDim.x - 1) {
        d_blockSums[blockIdx.x] = temp[pout * blockDim.x + tid];
    }
}

// После того как каждый блок вычислил свой локальный scan,
// нужно добавить суммы всех предыдущих блоков

__global__ void addBlockSums(float* d_output, float* d_blockSums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (blockIdx.x > 0 && idx < N) {
        // добавляем сумму всех предыдущих блоков
        d_output[idx] += d_blockSums[blockIdx.x - 1];
    }
}


void prefixSumCPU(float* input, float* output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; i++) {
        output[i] = output[i-1] + input[i];
    }
}

void scanBlockSumsRecursive(float* d_blockSums, int numBlocks) {
    if (numBlocks <= 1) return;
    
    // выделяем память для следующего уровня
    int nextNumBlocks = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* d_nextBlockSums;
    cudaMalloc((void**)&d_nextBlockSums, nextNumBlocks * sizeof(float));
    
    // сканируем блочные суммы
    float* d_tempOutput;
    cudaMalloc((void**)&d_tempOutput, numBlocks * sizeof(float));
    
    prefixScanShared<<<nextNumBlocks, BLOCK_SIZE>>>(d_blockSums, d_tempOutput, 
                                                     d_nextBlockSums, numBlocks);
    cudaDeviceSynchronize();
    
    // рекурсивно обрабатываем следующий уровень
    if (nextNumBlocks > 1) {
        scanBlockSumsRecursive(d_nextBlockSums, nextNumBlocks);
        addBlockSums<<<nextNumBlocks, BLOCK_SIZE>>>(d_tempOutput, d_nextBlockSums, numBlocks);
        cudaDeviceSynchronize();
    }
    
    // копируем результат обратно
    cudaMemcpy(d_blockSums, d_tempOutput, numBlocks * sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaFree(d_nextBlockSums);
    cudaFree(d_tempOutput);
}

// проверка корректности результата
bool verifyResult(float* cpu, float* gpu, int N, float tolerance = 0.01f) {
    for (int i = 0; i < N; i++) {
        if (fabs(cpu[i] - gpu[i]) > tolerance * fabs(cpu[i]) + tolerance) {
            cout << "Mismatch at index " << i << ": CPU=" << cpu[i] 
                 << ", GPU=" << gpu[i] << endl;
            return false;
        }
    }
    return true;
}

int main() {
    // размер массива по заданию
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    cout << "========================================" << endl;
    cout << " Prefix Sum (SCAN)" << endl;
    cout << " CUDA Shared Memory vs CPU" << endl;
    cout << "========================================" << endl << endl;
    
    cout << "parameters:" << endl;
    cout << "  array size: " << N << " elements" << endl;
    cout << "  memory: " << size / 1024.0 / 1024.0 << " MB" << endl;
    cout << "  block size: " << BLOCK_SIZE << endl << endl;
    
    // выделяем память на host
    float* h_input = new float[N];
    float* h_outputCPU = new float[N];
    float* h_outputGPU = new float[N];
    
    // инициализация массива
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 10); 
    }
    
    // выводим первые элементы
    cout << "first 10 elements input: ";
    for (int i = 0; i < 10; i++) {
        cout << h_input[i] << " ";
    }
    cout << endl << endl;
    
    
    cout << "========================================" << endl;
    cout << " CPU" << endl;
    cout << "========================================" << endl;
    
    auto cpuStart = high_resolution_clock::now();
    
    prefixSumCPU(h_input, h_outputCPU, N);
    
    auto cpuEnd = high_resolution_clock::now();
    duration<double, milli> cpuTime = cpuEnd - cpuStart;
    
    cout << "first 10 elements output: ";
    for (int i = 0; i < 10; i++) {
        cout << h_outputCPU[i] << " ";
    }
    cout << endl;
    cout << "time: " << fixed << setprecision(3) << cpuTime.count() << " ms" << endl << endl;
    
    // ============ GPU ВЫЧИСЛЕНИЯ ============
    
    cout << "========================================" << endl;
    cout << " GPU (CUDA Shared Memory)" << endl;
    cout << "========================================" << endl;
    
    // конфигурация
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout << "numblocks: " << numBlocks << endl;
    
    // выделяем память на GPU
    float *d_input, *d_output, *d_blockSums;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(float));
    
    // копируем данные на GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // создаём CUDA events для точного замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // замер времени
    auto gpuStartTotal = high_resolution_clock::now();
    cudaEventRecord(start);
    
    // ЭТАП 1: локальный scan в каждом блоке
    prefixScanShared<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, N);
    cudaDeviceSynchronize();
    
    // ЭТАП 2: scan блочных сумм (рекурсивно)
    scanBlockSumsRecursive(d_blockSums, numBlocks);
    
    // ЭТАП 3: добавляем блочные суммы к результатам
    addBlockSums<<<numBlocks, BLOCK_SIZE>>>(d_output, d_blockSums, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    auto gpuEndTotal = high_resolution_clock::now();
    duration<double, milli> gpuTimeTotal = gpuEndTotal - gpuStartTotal;
    
    // получаем время из CUDA events
    float gpuTimeMs;
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    
    // копируем результат
    cudaMemcpy(h_outputGPU, d_output, size, cudaMemcpyDeviceToHost);
    
    cout << "first 10 elements output: ";
    for (int i = 0; i < 10; i++) {
        cout << h_outputGPU[i] << " ";
    }
    cout << endl;
    cout << "time (CUDA events): " << fixed << setprecision(3) << gpuTimeMs << " ms" << endl;
    cout << "time (wall clock): " << fixed << setprecision(3) << gpuTimeTotal.count() << " ms" << endl << endl;
    
    // ============ VERIFICATION AND COMPARISON ============
    
    cout << "========================================" << endl;
    cout << " RESULT COMPARISON" << endl;
    cout << "========================================" << endl << endl;
    
    // проверяем корректность
    bool correct = verifyResult(h_outputCPU, h_outputGPU, N);
    cout << "correctness: " << (correct ? "[OK]" : "[FAIL]") << endl << endl;
    
    // проверяем несколько позиций
    cout << "value comparison:" << endl;
    int checkPoints[] = {0, 10, 100, 1000, 10000, 100000, 999999};
    cout << "+----------+----------------+----------------+" << endl;
    cout << "| Index    | CPU            | GPU            |" << endl;
    cout << "+----------+----------------+----------------+" << endl;
    for (int idx : checkPoints) {
        if (idx < N) {
            cout << "| " << setw(8) << idx 
                 << " | " << setw(14) << fixed << setprecision(1) << h_outputCPU[idx]
                 << " | " << setw(14) << fixed << setprecision(1) << h_outputGPU[idx] << " |" << endl;
        }
    }
    cout << "+----------+----------------+----------------+" << endl << endl;
    
    // таблица производительности
    cout << "+------------------+------------+------------+" << endl;
    cout << "| Method            | Time (ms)  | Speedup    |" << endl;
    cout << "+------------------+------------+------------+" << endl;
    cout << "| CPU (sequential)  | " << setw(10) << fixed << setprecision(3) << cpuTime.count() 
         << " | " << setw(10) << "1.00x" << " |" << endl;
    cout << "| GPU (shared mem) | " << setw(10) << fixed << setprecision(3) << gpuTimeMs 
         << " | " << setw(9) << fixed << setprecision(2) << cpuTime.count() / gpuTimeMs << "x |" << endl;
    cout << "+------------------+------------+------------+" << endl << endl;
    
    // текстовый график
    cout << "speed:" << endl;
    double maxTime = max(cpuTime.count(), (double)gpuTimeMs);
    
    cout << "  CPU |";
    int cpuBar = (int)((cpuTime.count() / maxTime) * 40);
    for (int i = 0; i < cpuBar; i++);
    cout << " " << fixed << setprecision(2) << cpuTime.count() << " ms" << endl;
    
    cout << "  GPU |";
    int gpuBar = (int)((gpuTimeMs / maxTime) * 40);
    for (int i = 0; i < gpuBar; i++);
    cout << " " << fixed << setprecision(2) << gpuTimeMs << " ms" << endl;
    cout << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
    delete[] h_input;
    delete[] h_outputCPU;
    delete[] h_outputGPU;
    
    return 0;
}
