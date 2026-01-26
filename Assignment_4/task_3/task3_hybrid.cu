#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <thread>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// CUDA kernel для обработки части массива на GPU
__global__ void processArrayGPU(float* d_input, float* d_output, float* d_partial, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // вычисляем квадрат и загружаем в shared memory
    float val = 0.0f;
    if (idx < N) {
        val = d_input[idx] * d_input[idx]; 
        d_output[idx] = val;
    }
    sdata[tid] = val;
    __syncthreads();
    
    // редукция суммы в shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // записываем частичную сумму блока
    if (tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

// CPU функция для обработки части массива
// Возвращает сумму квадратов
float processArrayCPU(float* input, float* output, int start, int end) {
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        output[i] = input[i] * input[i]; 
        sum += output[i];
    }
    return sum;
}

// функция для CPU обработки в отдельном потоке (для гибридного режима)
void cpuWorker(float* input, float* output, int start, int end, float* resultSum) {
    *resultSum = processArrayCPU(input, output, start, end);
}

// финальная редукция на CPU
float finalReduce(float* partialSums, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += partialSums[i];
    }
    return sum;
}

int main() {
    int N = 10000000;  
    size_t size = N * sizeof(float);
    
    cout << "========================================" << endl;
    cout << " CPU + GPU" << endl;
    cout << "========================================" << endl << endl;
    
    cout << "config:" << endl;
    cout << "  array size: " << N << " elements" << endl;
    cout << "  memory: " << size / 1024.0 / 1024.0 << " MB" << endl;
    cout << "  operation: square elements + sum" << endl << endl;
    
    // выделяем память на host
    float* h_input = new float[N];
    float* h_output = new float[N];
    float* h_outputCPU = new float[N];
    float* h_outputGPU = new float[N];
    float* h_outputHybrid = new float[N];
    
    // инициализация массива
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;  // от 0.0 до 1.0
    }
    
    cout << "first 5 elements input: ";
    for (int i = 0; i < 5; i++) cout << fixed << setprecision(2) << h_input[i] << " ";
    cout << endl << endl;
    
    cout << "========================================" << endl;
    cout << " TEST 1: CPU Only" << endl;
    cout << "========================================" << endl;
    
    auto cpuStart = high_resolution_clock::now();
    
    float cpuSum = processArrayCPU(h_input, h_outputCPU, 0, N);
    
    auto cpuEnd = high_resolution_clock::now();
    duration<double, milli> cpuTime = cpuEnd - cpuStart;
    
    cout << "  sum of squares: " << fixed << setprecision(2) << cpuSum << endl;
    cout << "  time: " << fixed << setprecision(3) << cpuTime.count() << " ms" << endl << endl;
    
    // ============================================================
    // TEST 2: GPU Only 
    // ============================================================
    
    cout << "========================================" << endl;
    cout << " TEST 2: GPU Only" << endl;
    cout << "========================================" << endl;
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    cout << "  blocks: " << numBlocks << ", threads/block: " << blockSize << endl;
    
    // allocate memory on GPU
    float *d_input, *d_output, *d_partial;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_partial, numBlocks * sizeof(float));
    
    float* h_partial = new float[numBlocks];
    
    auto gpuStart = high_resolution_clock::now();
    
    // копируем данные на GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // запускаем kernel
    processArrayGPU<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_output, d_partial, N);
    cudaDeviceSynchronize();
    
    // копируем результат
    cudaMemcpy(h_outputGPU, d_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float gpuSum = finalReduce(h_partial, numBlocks);
    
    auto gpuEnd = high_resolution_clock::now();
    duration<double, milli> gpuTime = gpuEnd - gpuStart;
    
    cout << "  sum of squares: " << fixed << setprecision(2) << gpuSum << endl;
    cout << "  time: " << fixed << setprecision(3) << gpuTime.count() << " ms" << endl << endl;
    
    // ============================================================
    // TEST 3: Hybrid CPU + GPU
    // ============================================================
    
    cout << "========================================" << endl;
    cout << " TEST 3: Hybrid CPU + GPU" << endl;
    cout << "========================================" << endl;
    
    float cpuRatio = 0.3f;  
    int cpuPart = (int)(N * cpuRatio);
    int gpuPart = N - cpuPart;
    
    cout << "  CPU processes: " << cpuPart << " elements (" 
         << (int)(cpuRatio * 100) << "%)" << endl;
    cout << "  GPU processes: " << gpuPart << " elements (" 
         << (int)((1 - cpuRatio) * 100) << "%)" << endl << endl;
    
    int gpuNumBlocks = (gpuPart + blockSize - 1) / blockSize;
    
    auto hybridStart = high_resolution_clock::now();
    
    // ===== Параллельный запуск CPU и GPU =====
    
    float cpuPartSum = 0.0f;
    
    // запускаем CPU в отдельном потоке
    thread cpuThread(cpuWorker, h_input, h_outputHybrid, 0, cpuPart, &cpuPartSum);
    
    // одновременно работает GPU
    // копируем только GPU часть данных
    cudaMemcpy(d_input, h_input + cpuPart, gpuPart * sizeof(float), cudaMemcpyHostToDevice);
    
    // запускаем kernel для GPU части
    processArrayGPU<<<gpuNumBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_output, d_partial, gpuPart);
    
    // ждём завершения GPU
    cudaDeviceSynchronize();
    
    // копируем результат GPU
    cudaMemcpy(h_outputHybrid + cpuPart, d_output, gpuPart * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partial, d_partial, gpuNumBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float gpuPartSum = finalReduce(h_partial, gpuNumBlocks);
    
    // ждём завершения CPU потока
    cpuThread.join();
    
    float hybridSum = cpuPartSum + gpuPartSum;
    
    auto hybridEnd = high_resolution_clock::now();
    duration<double, milli> hybridTime = hybridEnd - hybridStart;
    
    cout << "  CPU sum: " << fixed << setprecision(2) << cpuPartSum << endl;
    cout << "  GPU sum: " << fixed << setprecision(2) << gpuPartSum << endl;
    cout << "  total sum: " << fixed << setprecision(2) << hybridSum << endl;
    cout << "  time: " << fixed << setprecision(3) << hybridTime.count() << " ms" << endl << endl;
    
    // проверка корректности
    auto rel = [](double a, double b){
        return fabs(a-b) / (fabs(a) + 1e-12);
    };

    bool cpuGpuMatch  = rel(cpuSum, gpuSum) < 0.02;    // 2%
    bool hybridMatch  = rel(cpuSum, hybridSum) < 0.02;

    
    cout << "validation :" << endl;
    cout << "  CPU vs GPU: " << (cpuGpuMatch ? "[OK]" : "[FAIL]") << endl;
    cout << "  CPU vs Hybrid: " << (hybridMatch ? "[OK]" : "[FAIL]") << endl << endl;
    
    // таблица времени
    cout << "+------------------+------------+------------+" << endl;
    cout << "| Method           | Time (ms)  | Speedup    |" << endl;
    cout << "+------------------+------------+------------+" << endl;
    cout << "| CPU only         | " << setw(10) << fixed << setprecision(3) << cpuTime.count() 
         << " | " << setw(10) << "1.00x" << " |" << endl;
    cout << "| GPU only         | " << setw(10) << fixed << setprecision(3) << gpuTime.count() 
         << " | " << setw(9) << fixed << setprecision(2) << cpuTime.count() / gpuTime.count() << "x |" << endl;
    cout << "| Hybrid CPU+GPU   | " << setw(10) << fixed << setprecision(3) << hybridTime.count() 
         << " | " << setw(9) << fixed << setprecision(2) << cpuTime.count() / hybridTime.count() << "x |" << endl;
    cout << "+------------------+------------+------------+" << endl << endl;
    
    // текстовый график
    cout << "performance:" << endl;
    double maxTime = std::max(cpuTime.count(), std::max(gpuTime.count(), hybridTime.count()));
    
    cout << "  CPU     |";
    int bar1 = (int)((cpuTime.count() / maxTime) * 40);
    for (int i = 0; i < bar1; i++);
    cout << " " << fixed << setprecision(2) << cpuTime.count() << " ms" << endl;
    
    cout << "  GPU     |";
    int bar2 = (int)((gpuTime.count() / maxTime) * 40);
    for (int i = 0; i < bar2; i++);
    cout << " " << fixed << setprecision(2) << gpuTime.count() << " ms" << endl;
    
    cout << "  Hybrid  |";
    int bar3 = (int)((hybridTime.count() / maxTime) * 40);
    for (int i = 0; i < bar3; i++);
    cout << " " << fixed << setprecision(2) << hybridTime.count() << " ms" << endl;
    cout << endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    delete[] h_input;
    delete[] h_output;
    delete[] h_outputCPU;
    delete[] h_outputGPU;
    delete[] h_outputHybrid;
    delete[] h_partial;
    
    return 0;
}
