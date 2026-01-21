/*
 * Лабораторная работа: Измерение производительности
 * Задача 4: Бенчмарк разных типов памяти
 * 
 * Измеряем время выполнения для массивов:
 * - 10,000 элементов
 * - 100,000 элементов
 * - 1,000,000 элементов
 * 
 * Сравниваем: Global Memory vs Shared Memory
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace std::chrono;

// ============================================================
// Операция: поэлементное возведение в квадрат и сумма
// Это даст нагрузку на память и вычисления
// ============================================================

// Вариант 1: Только глобальная память
__global__ void processGlobalOnly(float* d_input, float* d_output, float* d_partial, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    
    // обработка - читаем и пишем в глобальную память
    if (idx < N) {
        // возводим в квадрат (работа с global memory)
        float val = d_input[idx];
        d_output[idx] = val * val;
    }
    __syncthreads();
    
    // редукция суммы в глобальной памяти (медленно!)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (blockStart + tid + stride) < N) {
            d_output[blockStart + tid] += d_output[blockStart + tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0 && blockStart < N) {
        d_partial[blockIdx.x] = d_output[blockStart];
    }
}

// Вариант 2: Глобальная + Shared память
__global__ void processSharedMemory(float* d_input, float* d_partial, int N) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // загружаем в shared memory и сразу обрабатываем
    if (idx < N) {
        float val = d_input[idx];
        sdata[tid] = val * val;  // возводим в квадрат
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // редукция в shared memory (быстро!)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

// Вариант 3: Оптимизированная версия (2 элемента на поток + warp unroll)
__global__ void processOptimized(float* d_input, float* d_partial, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // каждый поток обрабатывает 2 элемента
    float sum = 0.0f;
    if (idx < N) {
        float val = d_input[idx];
        sum = val * val;
    }
    if (idx + blockDim.x < N) {
        float val = d_input[idx + blockDim.x];
        sum += val * val;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // редукция с развернутым warp
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // последний warp без синхронизации
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

// структура для результатов теста
struct BenchmarkResult {
    double globalTime;
    double sharedTime;
    double optimizedTime;
    float globalSum;
    float sharedSum;
    float optimizedSum;
};

// функция бенчмарка для одного размера
BenchmarkResult runBenchmark(int N, int numRuns = 5) {
    BenchmarkResult result = {0, 0, 0, 0, 0, 0};
    
    size_t size = N * sizeof(float);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int numBlocksOpt = (N + blockSize * 2 - 1) / (blockSize * 2);
    
    // память на host
    float* h_input = new float[N];
    float* h_partial = new float[numBlocks];
    
    // инициализация данных
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;  // маленькие числа
    }
    
    // память на GPU
    float *d_input, *d_output, *d_partial;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_partial, numBlocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // прогрев GPU
    processSharedMemory<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_partial, N);
    cudaDeviceSynchronize();
    
    // ============ ТЕСТ 1: Global Memory Only ============
    double globalTotal = 0;
    for (int run = 0; run < numRuns; run++) {
        auto start = high_resolution_clock::now();
        
        processGlobalOnly<<<numBlocks, blockSize>>>(d_input, d_output, d_partial, N);
        cudaDeviceSynchronize();
        
        auto end = high_resolution_clock::now();
        duration<double, milli> elapsed = end - start;
        globalTotal += elapsed.count();
    }
    result.globalTime = globalTotal / numRuns;
    
    // получаем результат
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    result.globalSum = 0;
    for (int i = 0; i < numBlocks; i++) result.globalSum += h_partial[i];
    
    // ============ ТЕСТ 2: Shared Memory ============
    double sharedTotal = 0;
    for (int run = 0; run < numRuns; run++) {
        auto start = high_resolution_clock::now();
        
        processSharedMemory<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_partial, N);
        cudaDeviceSynchronize();
        
        auto end = high_resolution_clock::now();
        duration<double, milli> elapsed = end - start;
        sharedTotal += elapsed.count();
    }
    result.sharedTime = sharedTotal / numRuns;
    
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    result.sharedSum = 0;
    for (int i = 0; i < numBlocks; i++) result.sharedSum += h_partial[i];
    
    // ============ ТЕСТ 3: Optimized ============
    double optTotal = 0;
    for (int run = 0; run < numRuns; run++) {
        auto start = high_resolution_clock::now();
        
        processOptimized<<<numBlocksOpt, blockSize, blockSize * sizeof(float)>>>(d_input, d_partial, N);
        cudaDeviceSynchronize();
        
        auto end = high_resolution_clock::now();
        duration<double, milli> elapsed = end - start;
        optTotal += elapsed.count();
    }
    result.optimizedTime = optTotal / numRuns;
    
    cudaMemcpy(h_partial, d_partial, numBlocksOpt * sizeof(float), cudaMemcpyDeviceToHost);
    result.optimizedSum = 0;
    for (int i = 0; i < numBlocksOpt; i++) result.optimizedSum += h_partial[i];
    
    // освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    delete[] h_input;
    delete[] h_partial;
    
    return result;
}

// функция для рисования текстового графика
void drawBarChart(const char* title, double values[], const char* labels[], int count, double maxVal) {
    cout << title << endl;
    cout << string(60, '-') << endl;
    
    for (int i = 0; i < count; i++) {
        int barLength = (int)((values[i] / maxVal) * 40);
        cout << setw(12) << labels[i] << " | ";
        for (int j = 0; j < barLength; j++) cout << "█";
        cout << " " << fixed << setprecision(3) << values[i] << " ms" << endl;
    }
    cout << string(60, '-') << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << " БЕНЧМАРК: сравнение типов памяти CUDA" << endl;
    cout << "========================================" << endl << endl;
    
    // получаем информацию о GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "GPU: " << prop.name << endl;
    cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "Global Memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << endl;
    cout << endl;
    
    // размеры для тестирования
    int sizes[] = {10000, 100000, 1000000};
    const char* sizeLabels[] = {"10K", "100K", "1M"};
    int numSizes = 3;
    
    // массивы для хранения результатов
    BenchmarkResult results[3];
    
    // запускаем тесты
    cout << "Запуск бенчмарков (5 прогонов на каждый тест)..." << endl << endl;
    
    for (int i = 0; i < numSizes; i++) {
        cout << "Тестирование N = " << sizes[i] << "..." << endl;
        results[i] = runBenchmark(sizes[i]);
        
        cout << "  Global:    " << fixed << setprecision(3) << results[i].globalTime << " ms" << endl;
        cout << "  Shared:    " << fixed << setprecision(3) << results[i].sharedTime << " ms" << endl;
        cout << "  Optimized: " << fixed << setprecision(3) << results[i].optimizedTime << " ms" << endl;
        cout << endl;
    }
    
    // ============ ВЫВОД РЕЗУЛЬТАТОВ ============
    cout << "========================================" << endl;
    cout << " РЕЗУЛЬТАТЫ" << endl;
    cout << "========================================" << endl << endl;
    
    // таблица результатов
    cout << "+----------+------------+------------+------------+----------+----------+" << endl;
    cout << "| Размер   | Global(ms) | Shared(ms) | Optim.(ms) | Shared/G | Optim/G  |" << endl;
    cout << "+----------+------------+------------+------------+----------+----------+" << endl;
    
    for (int i = 0; i < numSizes; i++) {
        double speedupShared = results[i].globalTime / results[i].sharedTime;
        double speedupOpt = results[i].globalTime / results[i].optimizedTime;
        
        cout << "| " << setw(8) << sizeLabels[i] 
             << " | " << setw(10) << fixed << setprecision(3) << results[i].globalTime
             << " | " << setw(10) << fixed << setprecision(3) << results[i].sharedTime
             << " | " << setw(10) << fixed << setprecision(3) << results[i].optimizedTime
             << " | " << setw(7) << fixed << setprecision(2) << speedupShared << "x"
             << " | " << setw(7) << fixed << setprecision(2) << speedupOpt << "x |" << endl;
    }
    cout << "+----------+------------+------------+------------+----------+----------+" << endl;
    cout << endl;
    
    // графики
    cout << "========================================" << endl;
    cout << " ГРАФИКИ" << endl;
    cout << "========================================" << endl << endl;
    
    // находим максимальное значение для масштабирования
    double maxTime = 0;
    for (int i = 0; i < numSizes; i++) {
        if (results[i].globalTime > maxTime) maxTime = results[i].globalTime;
    }
    
    // график для каждого размера
    for (int i = 0; i < numSizes; i++) {
        cout << "N = " << sizes[i] << " элементов:" << endl;
        cout << "  Global    |";
        int bar1 = (int)((results[i].globalTime / maxTime) * 40);
        for (int j = 0; j < bar1; j++) cout << "█";
        cout << " " << fixed << setprecision(3) << results[i].globalTime << " ms" << endl;
        
        cout << "  Shared    |";
        int bar2 = (int)((results[i].sharedTime / maxTime) * 40);
        for (int j = 0; j < bar2; j++) cout << "▓";
        cout << " " << fixed << setprecision(3) << results[i].sharedTime << " ms" << endl;
        
        cout << "  Optimized |";
        int bar3 = (int)((results[i].optimizedTime / maxTime) * 40);
        for (int j = 0; j < bar3; j++) cout << "░";
        cout << " " << fixed << setprecision(3) << results[i].optimizedTime << " ms" << endl;
        cout << endl;
    }
    
    // график зависимости от размера
    cout << "========================================" << endl;
    cout << " ЗАВИСИМОСТЬ ВРЕМЕНИ ОТ РАЗМЕРА МАССИВА" << endl;
    cout << "========================================" << endl << endl;
    
    cout << "Global Memory:" << endl;
    cout << "  10K   |";
    for (int j = 0; j < (int)((results[0].globalTime / maxTime) * 50); j++) cout << "█";
    cout << endl;
    cout << "  100K  |";
    for (int j = 0; j < (int)((results[1].globalTime / maxTime) * 50); j++) cout << "█";
    cout << endl;
    cout << "  1M    |";
    for (int j = 0; j < (int)((results[2].globalTime / maxTime) * 50); j++) cout << "█";
    cout << endl << endl;
    
    cout << "Shared Memory:" << endl;
    cout << "  10K   |";
    for (int j = 0; j < (int)((results[0].sharedTime / maxTime) * 50); j++) cout << "▓";
    cout << endl;
    cout << "  100K  |";
    for (int j = 0; j < (int)((results[1].sharedTime / maxTime) * 50); j++) cout << "▓";
    cout << endl;
    cout << "  1M    |";
    for (int j = 0; j < (int)((results[2].sharedTime / maxTime) * 50); j++) cout << "▓";
    cout << endl << endl;
    
    // выводы
    cout << "========================================" << endl;
    cout << " ВЫВОДЫ" << endl;
    cout << "========================================" << endl << endl;
    
    cout << "1. SHARED MEMORY значительно быстрее GLOBAL MEMORY" << endl;
    cout << "   - Латентность: 5-10 cycles vs 400-800 cycles" << endl;
    cout << "   - Ускорение: " << fixed << setprecision(1) 
         << results[2].globalTime / results[2].sharedTime << "x для 1M элементов" << endl << endl;
    
    cout << "2. ОПТИМИЗАЦИЯ (2 элемента на поток + warp unroll) даёт" << endl;
    cout << "   дополнительное ускорение: " << fixed << setprecision(1)
         << results[2].sharedTime / results[2].optimizedTime << "x" << endl << endl;
    
    cout << "3. С ростом размера массива преимущество Shared Memory растёт:" << endl;
    for (int i = 0; i < numSizes; i++) {
        cout << "   - " << sizeLabels[i] << ": " << fixed << setprecision(2)
             << results[i].globalTime / results[i].sharedTime << "x" << endl;
    }
    cout << endl;
    
    cout << "4. Рекомендации:" << endl;
    cout << "   - Всегда использовать Shared Memory для промежуточных вычислений" << endl;
    cout << "   - Минимизировать обращения к Global Memory" << endl;
    cout << "   - Использовать coalesced доступ к памяти" << endl;
    cout << "   - Разворачивать последний warp для избежания __syncthreads()" << endl;
    
    return 0;
}
