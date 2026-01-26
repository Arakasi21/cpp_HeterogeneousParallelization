/*
 * Лабораторная работа: Гибридные и распределённые параллельные вычисления
 * Задание 1: Сумма элементов массива (CUDA vs CPU)
 * 
 * Цель: реализовать редукцию суммы на GPU с глобальной памятью
 * и сравнить с последовательной реализацией на CPU
 * Размер массива: 100,000 элементов
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace std::chrono;

// Каждый блок вычисляет частичную сумму своего участка массива

__global__ void sumReductionGlobal(float* d_input, float* d_output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    int blockStart = blockIdx.x * blockDim.x;
    
    if (idx >= N) return;
    
    __syncthreads();
    
    // на каждой итерации половина потоков складывает пары элементов
    // stride - расстояние между складываемыми элементами
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // проверяем границы массива
            int partnerIdx = blockStart + tid + stride;
            if (partnerIdx < N) {
                d_input[blockStart + tid] += d_input[partnerIdx];
            }
        }
        __syncthreads();
    }
    
    // первый поток блока записывает результат
    // это частичная сумма для данного блока
    if (tid == 0) {
        d_output[blockIdx.x] = d_input[blockStart];
    }
}


float sumCPU(float* array, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += array[i];
    }
    return sum;
}

// функция для финальной редукции частичных сумм на CPU
float finalReduceCPU(float* partialSums, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; i++) {
        total += partialSums[i];
    }
    return total;
}

int main() {
    int N = 100000;
    size_t size = N * sizeof(float);
    cout << "config:" << endl;
    cout << "  array size: " << N << " elements" << endl;
    cout << "  memory: " << size / 1024.0 << " KB" << endl << endl;
    
    
    // память на host (CPU)
    float* h_input = new float[N];      
    float* h_inputCopy = new float[N]; 
    
    srand(42);  // фиксированный seed для воспроизводимости
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;  // от 0.0 до 10.0
        h_inputCopy[i] = h_input[i];
    }
    
    cout << "first 10 elements: ";
    for (int i = 0; i < 10; i++) {
        cout << fixed << setprecision(1) << h_input[i] << " ";
    }
    cout << endl << endl;
    
    
    cout << " CPU" << endl;
    
    // замеряем время CPU
    auto cpuStart = high_resolution_clock::now();
    
    float cpuSum = sumCPU(h_input, N);
    
    auto cpuEnd = high_resolution_clock::now();
    duration<double, milli> cpuTime = cpuEnd - cpuStart;
    
    cout << "  result: " << fixed << setprecision(2) << cpuSum << endl;
    cout << "  time: " << fixed << setprecision(4) << cpuTime.count() << " ms" << endl << endl;
    
    
    cout << " cuda global memory" << endl;
    
    // конфигурация запуска
    int blockSize = 256;  // потоков в блоке
    int numBlocks = (N + blockSize - 1) / blockSize;  // количество блоков
    
    cout << "  block size: " << blockSize << endl;
    cout << "  number of blocks: " << numBlocks << endl;
    cout << "  total threads: " << numBlocks * blockSize << endl << endl;
    
    // выделяем память на GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, numBlocks * sizeof(float));
    
    // массив для частичных сумм на host
    float* h_partial = new float[numBlocks];
    
    // копируем данные на GPU
    cudaMemcpy(d_input, h_inputCopy, size, cudaMemcpyHostToDevice);
    
    // замеряем время GPU 
    auto gpuStart = high_resolution_clock::now();
    
    // запускаем kernel
    sumReductionGlobal<<<numBlocks, blockSize>>>(d_input, d_output, N);
    
    // проверяем ошибки
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    
    // ждём завершения GPU
    cudaDeviceSynchronize();
    
    // копируем частичные суммы на host
    cudaMemcpy(h_partial, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // финальная редукция на CPU
    float gpuSum = finalReduceCPU(h_partial, numBlocks);
    
    auto gpuEnd = high_resolution_clock::now();
    duration<double, milli> gpuTime = gpuEnd - gpuStart;
    
    cout << "  result: " << fixed << setprecision(2) << gpuSum << endl;
    cout << "  time: " << fixed << setprecision(4) << gpuTime.count() << " ms" << endl << endl;
    
    // ============ RESULTS COMPARISON   ============
    
    cout << "========================================" << endl;
    cout << " COMPARISON" << endl;
    cout << "========================================" << endl << endl;
    
    // проверяем корректность
    float diff = fabs(cpuSum - gpuSum);
    float tolerance = cpuSum * 0.001f;  
    bool correct = diff < tolerance;
    
    cout << "correctness check:" << endl;
    cout << "  CPU sum: " << fixed << setprecision(2) << cpuSum << endl;
    cout << "  GPU sum: " << fixed << setprecision(2) << gpuSum << endl;
    cout << "  difference: " << fixed << setprecision(4) << diff << endl;
    cout << "  status: " << (correct ? "[OK]" : "[FAIL]") << endl << endl;
    
    // comparison table
    cout << "+----------------+------------+------------+" << endl;
    cout << "| Method         | Time (ms)  | Speedup    |" << endl;
    cout << "+----------------+------------+------------+" << endl;
    cout << "| CPU (serial)   | " << setw(10) << fixed << setprecision(4) << cpuTime.count() 
         << " | " << setw(10) << "1.00x" << " |" << endl;
    cout << "| GPU (global)   | " << setw(10) << fixed << setprecision(4) << gpuTime.count() 
         << " | " << setw(9) << fixed << setprecision(2) << cpuTime.count() / gpuTime.count() << "x |" << endl;
    cout << "+----------------+------------+------------+" << endl << endl;
    
    // text-based graph
    cout << "performance:" << endl;
    double maxTime = max(cpuTime.count(), gpuTime.count());
    
    cout << "  CPU |";
    int cpuBar = (int)((cpuTime.count() / maxTime) * 40);
    cout << " " << fixed << setprecision(3) << cpuTime.count() << " ms" << endl;
    
    cout << "  GPU |";
    int gpuBar = (int)((gpuTime.count() / maxTime) * 40);
    cout << " " << fixed << setprecision(3) << gpuTime.count() << " ms" << endl;
    cout << endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_inputCopy;
    delete[] h_partial;
    
    return 0;
}
