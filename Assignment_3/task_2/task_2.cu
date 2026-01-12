#include <iostream> // cin cout
#include <cuda_runtime.h> // для CUDA функций
#include <chrono> // high resolution clock для замера времени
#include <iomanip> // setprecision, fixed

using namespace std;
using namespace std::chrono;

// CUDA kernel для поэлементного сложения
// __global__ - функция выполняется на GPU
// d_a, d_b - входные массивы на GPU
// d_c - выходной массив на GPU (результат A + B)
// N - размер массивов
__global__ void vectorAdd(float* d_a, float* d_b, float* d_c, int N) {
    // вычисляем глобальный индекс потока
    // blockIdx.x - индекс блока в сетке
    // blockDim.x - количество потоков в блоке (это и есть размер блока который мы будем менять!)
    // threadIdx.x - индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы массива
    // нужно потому что общее кол-во потоков может быть больше чем N
    if (idx < N) {
        // простое сложение элементов
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

// функция для тестирования с определенным размером блока
// возвращает время выполнения в секундах
double testWithBlockSize(float* d_a, float* d_b, float* d_c, float* h_c, int N, int blockSize) {
    // вычисляем количество блоков
    // формула округления вверх: (N + blockSize - 1) / blockSize
    // например: N=1000, blockSize=256 -> (1000+255)/256 = 1255/256 = 4 блока
    int blocksPerGrid = (N + blockSize - 1) / blockSize;
    
    cout << "  Block size: " << blockSize << " threads" << endl;
    cout << "  Blocks: " << blocksPerGrid << endl;
    cout << "  Total threads: " << blocksPerGrid * blockSize << endl;
    
    // замеряем время
    auto start = high_resolution_clock::now();
    
    // запускаем kernel с указанным размером блока
    // <<<blocksPerGrid, blockSize>>> - конфигурация запуска
    vectorAdd<<<blocksPerGrid, blockSize>>>(d_a, d_b, d_c, N);
    
    // проверяем ошибки CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "  CUDA Error: " << cudaGetErrorString(err) << endl;
        return -1.0;
    }
    
    // ждем завершения GPU
    cudaDeviceSynchronize();
    
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    
    // копируем результат обратно на host для проверки
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cout << "  Time: " << fixed << setprecision(6) << elapsed.count() << " sec" << endl;
    
    return elapsed.count();
}

int main() {
    int N = 100000000; 
    size_t size = N * sizeof(float); 
    
    cout << "n: " << N << " elements" << endl;
    cout << "memory: " << size / 1024 / 1024 << " MB" << endl;
    cout << "total memory: " << 3 * size / 1024 / 1024 << " MB" << endl << endl;
    
    // выделяем память на host (CPU)
    // h_ префикс = host memory
    float* h_a = new float[N]; // массив A
    float* h_b = new float[N]; // массив B
    float* h_c = new float[N]; // результат C = A + B
    
    srand(time(0));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 100) / 10.0f; // от 0.0 до 10.0
        h_b[i] = (float)(rand() % 100) / 10.0f;
    }
    
    cout << "first 5 elements:" << endl;
    cout << "A: ";
    for (int i = 0; i < 5; i++) cout << h_a[i] << " ";
    cout << endl << "B: ";
    for (int i = 0; i < 5; i++) cout << h_b[i] << " ";
    cout << endl << endl;
    
    // выделяем память на GPU
    // d_ префикс = device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // копируем входные данные на GPU
    cout << "copying data to GPU" << endl;
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cout << endl;
    
    // массив размеров блоков для тестирования
    // разные размеры: от маленьких до максимальных
    // максимум 1024 потока на блок на большинстве GPU
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numTests = 6;
    
    // массив для хранения времен выполнения
    double times[6];
    
    // тестируем каждый размер блока
    for (int i = 0; i < numTests; i++) {
        cout << "test #" << (i + 1) << ":" << endl;
        times[i] = testWithBlockSize(d_a, d_b, d_c, h_c, N, blockSizes[i]);
        
        // проверяем корректность первых 5 результатов
        cout << "  result: ";
        bool correct = true;
        for (int j = 0; j < 5; j++) {
            float expected = h_a[j] + h_b[j];
            float got = h_c[j];
            cout << got << " ";
            if (abs(expected - got) > 0.001f) {
                correct = false;
            }
        }
        cout << (correct ? "[OK]" : "[FAIL]") << endl << endl;
    }
    
    cout << "Analysis" << endl << endl;
    
    cout << "+------------+------------+----------------+" << endl;
    cout << "| Block Size | Time (sec) | Relative Speed |" << endl;
    cout << "+------------+------------+----------------+" << endl;
    
    double bestTime = times[0];
    int bestIdx = 0;
    for (int i = 1; i < numTests; i++) {
        if (times[i] > 0 && times[i] < bestTime) {
            bestTime = times[i];
            bestIdx = i;
        }
    }
    
    for (int i = 0; i < numTests; i++) {
        if (times[i] > 0) {
            double relSpeed = bestTime / times[i];
            cout << "| " << setw(10) << blockSizes[i] 
                 << " | " << fixed << setprecision(6) << setw(10) << times[i]
                 << " | " << fixed << setprecision(2) << setw(14) << relSpeed << "x |";
            if (i == bestIdx) cout << " <-- BEST";
            cout << endl;
        }
    }
    cout << "+------------+------------+----------------+" << endl << endl;
    
    // находим худшее время
    double worstTime = times[0];
    for (int i = 1; i < numTests; i++) {
        if (times[i] > worstTime) worstTime = times[i];
    }
    
    // выводим статистику
    cout << "best conf.: " << blockSizes[bestIdx] << " threads/block" << endl;
    cout << "best t: " << fixed << setprecision(6) << bestTime << " sec" << endl;
    cout << "worst t: " << fixed << setprecision(6) << worstTime << " sec" << endl;
    cout << "difference: " << fixed << setprecision(2) 
         << (worstTime / bestTime) << "x" << endl << endl;
    
    // график производительности (текстовый)
    cout << "performance graph:" << endl;
    cout << "-----------------------------------------------------------" << endl;
    for (int i = 0; i < numTests; i++) {
        if (times[i] > 0) {
            int barLength = (int)((bestTime / times[i]) * 50);
            cout << setw(5) << blockSizes[i] << " | ";
            for (int j = 0; j < barLength; j++) cout << "#";
            cout << " " << fixed << setprecision(2) << (bestTime / times[i]) << "x" << endl;
        }
    }
    cout << "-----------------------------------------------------------" << endl;
    
    // освобождаем память
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return 0;
}