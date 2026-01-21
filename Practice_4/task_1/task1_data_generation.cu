/*
 * Лабораторная работа: Генерация массива случайных чисел на GPU
 * Задача 1: Подготовка данных
 * 
 * Цель: сгенерировать массив из 1,000,000 случайных чисел
 * используя CUDA и библиотеку cuRAND
 */
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>  // для генерации случайных чисел на GPU
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// CUDA kernel для инициализации генератора случайных чисел
// каждый поток получает свой уникальный seed
// state - массив состояний генератора (по одному на поток)
// seed - начальное значение для генератора
// N - количество элементов
__global__ void initRandom(curandState* state, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // seed - общий seed
        // idx - sequence number (чтобы каждый поток генерировал разные числа)
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// state - состояния генератора
// d_array - выходной массив на GPU
// N - размер массива
// minVal, maxVal - диапазон значений
__global__ void generateRandomNumbers(curandState* state, float* d_array, int N, 
                                       float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // curand_uniform генерирует число от 0.0 до 1.0
        float randomValue = curand_uniform(&state[idx]);
        d_array[idx] = minVal + randomValue * (maxVal - minVal);
    }
}

__global__ void generateRandomIntegers(curandState* state, int* d_array, int N,
                                        int minVal, int maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        unsigned int randomValue = curand(&state[idx]);
        d_array[idx] = minVal + (randomValue % (maxVal - minVal + 1));
    }
}

// функция для вывода статистики массива
void printArrayStats(float* h_array, int N) {
    float sum = 0.0f;
    float minVal = h_array[0];
    float maxVal = h_array[0];
    
    for (int i = 0; i < N; i++) {
        sum += h_array[i];
        if (h_array[i] < minVal) minVal = h_array[i];
        if (h_array[i] > maxVal) maxVal = h_array[i];
    }
    
    float avg = sum / N;
    
    cout << "array stats:" << endl;
    cout << "  min: " << fixed << setprecision(4) << minVal << endl;
    cout << "  max: " << fixed << setprecision(4) << maxVal << endl;
    cout << "  avg: " << fixed << setprecision(4) << avg << endl;
    cout << "  sum: " << fixed << setprecision(2) << sum << endl;
}

int main() {
    int N = 1000000;
    size_t sizeFloat = N * sizeof(float);
    size_t sizeState = N * sizeof(curandState);
    
    cout << "========================================" << endl;
    cout << " random number generation on GPU" << endl;
    cout << "========================================" << endl << endl;
    
    cout << "parameters:" << endl;
    cout << "  number of elements: " << N << endl;
    cout << "  memory for array: " << sizeFloat / 1024 / 1024 << " MB" << endl;
    cout << "  memory for generator: " << sizeState / 1024 / 1024 << " MB" << endl;
    cout << endl;
    
    // выделяем память на host
    float* h_array = new float[N];
    
    // выделяем память на GPU
    float* d_array;          // массив результатов
    curandState* d_states;   // состояния генератора
    
    cudaMalloc((void**)&d_array, sizeFloat);
    cudaMalloc((void**)&d_states, sizeState);
    
    // конфигурация запуска
    int blockSize = 256;
    int blocksPerGrid = (N + blockSize - 1) / blockSize;
    
    cout << " CUDA configuration:" << endl;
    cout << "  threads per block: " << blockSize << endl;
    cout << "  blocks per grid: " << blocksPerGrid << endl;
    cout << endl;
    
    // seed для генератора (используем текущее время)
    unsigned long seed = time(NULL);
    auto start1 = high_resolution_clock::now();
    
    initRandom<<<blocksPerGrid, blockSize>>>(d_states, seed, N);
    cudaDeviceSynchronize();
    
    auto end1 = high_resolution_clock::now();
    duration<double, milli> elapsed1 = end1 - start1;
    cout << "  initialization time: " << fixed << setprecision(2) 
         << elapsed1.count() << " ms" << endl << endl;    
    auto start2 = high_resolution_clock::now();
    
    generateRandomNumbers<<<blocksPerGrid, blockSize>>>(d_states, d_array, N, 0.0f, 100.0f);
    cudaDeviceSynchronize();
    
    auto end2 = high_resolution_clock::now();
    duration<double, milli> elapsed2 = end2 - start2;
    cout << "  generation time: " << fixed << setprecision(2) 
         << elapsed2.count() << " ms" << endl << endl;
    auto start3 = high_resolution_clock::now();
    
    cudaMemcpy(h_array, d_array, sizeFloat, cudaMemcpyDeviceToHost);
    
    auto end3 = high_resolution_clock::now();
    duration<double, milli> elapsed3 = end3 - start3;
    cout << "  copying time: " << fixed << setprecision(2) 
         << elapsed3.count() << " ms" << endl << endl;
    
    // выводим первые 10 элементов
    cout << "first 10 elements:" << endl << "  ";
    for (int i = 0; i < 10; i++) {
        cout << fixed << setprecision(2) << h_array[i] << " ";
    }
    cout << endl << endl;
    
    // выводим последние 10 элементов
    cout << "last 10 elements:" << endl << "  ";
    for (int i = N - 10; i < N; i++) {
        cout << fixed << setprecision(2) << h_array[i] << " ";
    }
    cout << endl << endl;
    
    // статистика
    printArrayStats(h_array, N);
    
    cout << endl;
    cout << "========================================" << endl;
    cout << " total time" << endl;
    cout << "========================================" << endl;
    cout << "  initialization: " << fixed << setprecision(2) << elapsed1.count() << " ms" << endl;
    cout << "  generation:     " << fixed << setprecision(2) << elapsed2.count() << " ms" << endl;
    cout << "  copying:       " << fixed << setprecision(2) << elapsed3.count() << " ms" << endl;
    cout << "  TOTAL:         " << fixed << setprecision(2) 
         << (elapsed1.count() + elapsed2.count() + elapsed3.count()) << " ms" << endl;
    
    // освобождаем память
    cudaFree(d_array);
    cudaFree(d_states);
    delete[] h_array;
    
    return 0;
}
