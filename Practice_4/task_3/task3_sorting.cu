/*
 * Лабораторная работа: Сортировка на GPU
 * Задача 3: Оптимизация сортировки
 * 
 * Реализуем:
 * - Сортировка пузырьком для подмассивов (локальная память)
 * - Глобальная память для общего массива
 * - Слияние отсортированных подмассивов (shared memory)
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace std::chrono;

#define SUBARRAY_SIZE 256

__global__ void bubbleSortKernel(int* d_array, int N, int subarraySize) {
    __shared__ int sdata[SUBARRAY_SIZE];
    
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * subarraySize;
    int idx = blockStart + tid;
    
    // подмассив в shared memory
    if (idx < N && tid < subarraySize) {
        sdata[tid] = d_array[idx];
    } else {
        sdata[tid] = INT_MAX; 
    }
    __syncthreads();
    
    //сортировка пузырьком в shared memory
    // используем odd-even transposition sort для параллелизма
    
    // количество элементов в этом подмассиве
    int count = min(subarraySize, N - blockStart);
    if (count <= 0) return;
    for (int phase = 0; phase < count; phase++) {
        // сравниваем пары (0,1), (2,3), (4,5) и тд
        if (phase % 2 == 0) {
            if (tid % 2 == 0 && tid + 1 < count) {
                if (sdata[tid] > sdata[tid + 1]) {
                    int temp = sdata[tid];
                    sdata[tid] = sdata[tid + 1];
                    sdata[tid + 1] = temp;
                }
            }
        }
        // сравниваем пары (1,2), (3,4), (5,6) (нечетные индексы)
        else {
            if (tid % 2 == 1 && tid + 1 < count) {
                if (sdata[tid] > sdata[tid + 1]) {
                    int temp = sdata[tid];
                    sdata[tid] = sdata[tid + 1];
                    sdata[tid + 1] = temp;
                }
            }
        }
        __syncthreads();
    }
    
    if (idx < N && tid < count) {
        d_array[idx] = sdata[tid];
    }
}

__global__ void mergeKernel(int* d_array, int* d_temp, int N, int width) {
    // shared memory для двух подмассивов
    extern __shared__ int smerge[];
    
    int tid = threadIdx.x;
    int mergeIdx = blockIdx.x;  // какую пару подмассивов сливаем
    
    // границы левого и правого подмассивов
    int left = mergeIdx * 2 * width;
    int mid = min(left + width, N);
    int right = min(left + 2 * width, N);
    
    if (left >= N) return;
    
    int leftSize = mid - left;
    int rightSize = right - mid;
    int totalSize = leftSize + rightSize;
    
    // загружаем оба подмассива в shared memory
    // левый подмассив: smerge[0..leftSize-1]
    // правый подмассив: smerge[width..width+rightSize-1]
    for (int i = tid; i < leftSize; i += blockDim.x) {
        smerge[i] = d_array[left + i];
    }
    for (int i = tid; i < rightSize; i += blockDim.x) {
        smerge[width + i] = d_array[mid + i];
    }
    __syncthreads();

    // используем binary search для определения позиции
    for (int i = tid; i < totalSize; i += blockDim.x) {
        int val;
        int fromLeft;  // элемент из левого или правого подмассива
        
        if (i < leftSize) {
            val = smerge[i];
            fromLeft = 1;
        } else {
            val = smerge[width + (i - leftSize)];
            fromLeft = 0;
        }
        
        int pos = 0;
        
        // считаем в левом подмассиве
        for (int j = 0; j < leftSize; j++) {
            if (smerge[j] < val || (smerge[j] == val && fromLeft == 0 && j < i)) {
                pos++;
            }
        }
        
        // считаем в правом подмассиве
        for (int j = 0; j < rightSize; j++) {
            if (smerge[width + j] < val || (smerge[width + j] == val && fromLeft == 1)) {
                pos++;
            }
        }
        
        // записываем в временный массив
        if (left + pos < N) {
            d_temp[left + pos] = val;
        }
    }
}
void mergeCPU(int* arr, int left, int mid, int right) {
    int n1 = mid - left;
    int n2 = right - mid;
    
    int* L = new int[n1];
    int* R = new int[n2];
    
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + i];
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    
    delete[] L;
    delete[] R;
}

bool isSorted(int* arr, int N) {
    for (int i = 1; i < N; i++) {
        if (arr[i] < arr[i-1]) return false;
    }
    return true;
}

void printArray(int* arr, int N, int count = 5) {
    cout << "first " << count << ": ";
    for (int i = 0; i < min(count, N); i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    
    cout << "last " << count << ": ";
    for (int i = max(0, N - count); i < N; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main() {
    // тестируем на разных размерах
    int sizes[] = {10000, 100000, 1000000};
    int numSizes = 3;

    cout << " bubble sort + merge" << endl;
    
    for (int s = 0; s < numSizes; s++) {
        int N = sizes[s];
        size_t size = N * sizeof(int);
        
        cout << "array size: " << N << " elements" << endl;
        
        int* h_array = new int[N];
        int* h_result = new int[N];
        int* h_reference = new int[N];
        
        srand(42);
        for (int i = 0; i < N; i++) {
            h_array[i] = rand() % 10000;
            h_reference[i] = h_array[i];
        }
        
        cout << "initial array:" << endl;
        printArray(h_array, N);
        cout << endl;
        auto cpuStart = high_resolution_clock::now();
        sort(h_reference, h_reference + N);
        auto cpuEnd = high_resolution_clock::now();
        duration<double, milli> cpuTime = cpuEnd - cpuStart;
        
        cout << "CPU: " << fixed << setprecision(2) 
             << cpuTime.count() << " ms" << endl << endl;
        
        int *d_array, *d_temp;
        cudaMalloc((void**)&d_array, size);
        cudaMalloc((void**)&d_temp, size);
        
        cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
        
        cout << "bubble sort" << endl;
        
        int subarraySize = SUBARRAY_SIZE;
        int numSubarrays = (N + subarraySize - 1) / subarraySize;
        
        cout << "  subarray size: " << subarraySize << endl;
        cout << "  number of subarrays: " << numSubarrays << endl;
        
        auto gpuStart = high_resolution_clock::now();
        
        bubbleSortKernel<<<numSubarrays, subarraySize>>>(d_array, N, subarraySize);
        cudaDeviceSynchronize();
        
        auto phase1End = high_resolution_clock::now();
        duration<double, milli> phase1Time = phase1End - gpuStart;
        cout << "  time: " << fixed << setprecision(2) << phase1Time.count() << " ms" << endl;
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cout << "  CUDA Error: " << cudaGetErrorString(err) << endl;
        }
        cout << endl;
        
        cout << "merge" << endl;
        
        int width = subarraySize;
        int mergeIterations = 0;
        
        while (width < N) {
            int numMerges = (N + 2 * width - 1) / (2 * width);
            size_t sharedSize = 2 * width * sizeof(int);
            
            // если shared memory достаточно - используем GPU
            // иначе делаем слияние на CPU
            if (sharedSize <= 48 * 1024) { 
                mergeKernel<<<numMerges, 256, sharedSize>>>(d_array, d_temp, N, width);
                cudaDeviceSynchronize();
                
                cudaMemcpy(d_array, d_temp, size, cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpy(h_result, d_array, size, cudaMemcpyDeviceToHost);
                
                for (int i = 0; i < N; i += 2 * width) {
                    int left = i;
                    int mid = min(i + width, N);
                    int right = min(i + 2 * width, N);
                    
                    if (mid < right) {
                        mergeCPU(h_result, left, mid, right);
                    }
                }
                
                cudaMemcpy(d_array, h_result, size, cudaMemcpyHostToDevice);
            }
            
            width *= 2;
            mergeIterations++;
        }
        
        auto phase2End = high_resolution_clock::now();
        duration<double, milli> phase2Time = phase2End - phase1End;
        cout << "  merge iterations: " << mergeIterations << endl;
        cout << "  time: " << fixed << setprecision(2) << phase2Time.count() << " ms" << endl;
        cout << endl;
 
        cudaMemcpy(h_result, d_array, size, cudaMemcpyDeviceToHost);
        
        auto gpuEnd = high_resolution_clock::now();
        duration<double, milli> gpuTotalTime = gpuEnd - gpuStart;
        
        cout << "RESULT:" << endl;
        printArray(h_result, N);
        
        bool sorted = isSorted(h_result, N);
        bool correct = true;
        for (int i = 0; i < N && correct; i++) {
            if (h_result[i] != h_reference[i]) correct = false;
        }
        
        cout << "sorted: " << (sorted ? "[OK]" : "[FAIL]") << endl;
        cout << "matches reference: " << (correct ? "[OK]" : "[FAIL]") << endl;
        cout << endl;
        
        // statistics
        cout << "TIME:" << endl;
        cout << "  GPU bubble sort: " << fixed << setprecision(2) << phase1Time.count() << " ms" << endl;
        cout << "  GPU/CPU merge:   " << fixed << setprecision(2) << phase2Time.count() << " ms" << endl;
        cout << "  GPU TOTAL:       " << fixed << setprecision(2) << gpuTotalTime.count() << " ms" << endl;
        cout << "  CPU std::sort:   " << fixed << setprecision(2) << cpuTime.count() << " ms" << endl;
        cout << endl;
        
        cudaFree(d_array);
        cudaFree(d_temp);
        delete[] h_array;
        delete[] h_result;
        delete[] h_reference;
    }
    
    // итоговая таблица
    cout << "| Array size| GPU Bubble | GPU Merge | CPU sort |" << endl;
    cout << "|-----------|------------|-----------|----------|" << endl;
    cout << "| 10,000    |   ~X ms    |  ~Y ms    | ~Z ms    |" << endl;
    cout << "| 100,000   |   ~X ms    |  ~Y ms    | ~Z ms    |" << endl;
    cout << "| 1,000,000 |   ~X ms    |  ~Y ms    | ~Z ms    |" << endl;
    cout << endl;
    
    return 0;
}
