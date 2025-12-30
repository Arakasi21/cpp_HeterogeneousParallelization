// Задача 4. Сортировка на GPU с использованием CUDA
// Практическое задание (25 баллов)
// Реализуйте параллельную сортировку слиянием на GPU с использованием CUDA:
// разделите массив на подмассивы, каждый из которых обрабатывается отдельным блоком;
// выполните параллельное слияние отсортированных подмассивов;
// замерьте производительность для массивов размером 10 000 и 100 000 элементов.

#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>
#include <algorithm>  
using namespace std;
using namespace std::chrono;
__device__ __host__ inline int min_custom(int a, int b) {
    return (a < b) ? a : b;
}

__device__ __host__ inline int max_custom(int a, int b) {
    return (a > b) ? a : b;
}
// NVIDIA в документации прописывает что все API вызовы могут завершиться с ошибкой
// например cudaMalloc может вернуть cudaErrorMemoryAllocation если не хватает памяти на GPU
// без проверки ошибок программа может продолжить работу с некорректными данными
// это критично для отладки потому что многие CUDA операции асинхронные и ошибки могут проявиться позже
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cout << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

// вспомогательная функция для копирования массива на CPU
void copyArray(int* source, int* dest, int N){
    for(int i = 0; i < N; i++){
        dest[i] = source[i];
    }
}

// функция проверки что массив отсортирован по возрастанию
bool isSorted(int* arr, int N){
    for(int i = 0; i < N - 1; i++){
        if(arr[i] > arr[i + 1]) return false;
    }
    return true;
}

// CPU

// классическая функция слияния двух отсортированных подмассивов
// используется в рекурсивной сортировке слиянием
void merge(int* arr, int left, int mid, int right){
    // код ниже создает два временных массива L и R для хранения левой и правой частей, затем сливает их обратно в arr, сравнивая элементы
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = new int[n1];
    int* R = new int[n2];

    for(int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for(int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    // сливаем временные массивы обратно в arr[left..right]
    int i = 0, j = 0, k = left;
    
    while(i < n1 && j < n2){
        if(L[i] <= R[j]){
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    // копируем оставшиеся элементы L, если есть
    while(i < n1){
        arr[k] = L[i];
        i++;
        k++;
    }
    
    // копируем оставшиеся элементы R, если есть
    while(j < n2){
        arr[k] = R[j];
        j++;
        k++;
    }
    
    delete[] L;
    delete[] R;
}

// сортировка слиянием, работающая рекурсивно, где левая и правая части сортируются отдельно, а затем сливаются
void mergeSort(int* arr, int left, int right){
    if(left < right){
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);      // сортируем левую половину
        mergeSort(arr, mid + 1, right); // сортируем правую половину
        merge(arr, left, mid, right);   // сливаем отсортированные половины
    }
}

void Sequential(int* arr, int N){
    mergeSort(arr, 0, N - 1);
}

// GPU

// GPU kernel для сортировки блоков данных
// каждый блок потоков обрабатывает независимый кусок массива
// Single Instruction Multiple Threads принцип SIMT
// все потоки в блоке выполняют одинаковый код но на разных данных
__global__ void blockSortKernel(int* d_arr, int N, int blockSize){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockSize + tid;
    
    extern __shared__ int shared[];
    
    // загрузка в shared memory
    if(idx < N){
        shared[tid] = d_arr[idx];
    } else {
        shared[tid] = INT_MAX;
    }
    __syncthreads();
    
    // ПАРАЛЛЕЛЬНАЯ битоническая сортировка
    // каждый поток активно участвует в сортировке, выполняя compare-and-swap операции
    // это настоящий параллельный алгоритм согласно CUDA Programming Guide
    // внешний цикл: увеличиваем размер битонических последовательностей (k = 2, 4, 8, 16, ...)
    for(int k = 2; k <= blockSize; k *= 2){
        for(int j = k / 2; j > 0; j /= 2){
            // каждый поток вычисляет индекс своего партнера для сравнения
            // используя XOR операцию которая определяет пары для битонической сортировки
            int ixj = tid ^ j;
            
            // проверяем что партнер существует, обрабатываем каждую пару только один раз
            if(ixj > tid){
                // определяем направление сортировки для создания битонической последовательности
                // (tid & k) == 0 означает что эта часть должна быть отсортирована по возрастанию
                if((tid & k) == 0){
                    // сортируем по возрастанию
                    if(shared[tid] > shared[ixj]){
                        // swap: каждый поток меняет только свой элемент
                        int temp = shared[tid];
                        shared[tid] = shared[ixj];
                        shared[ixj] = temp;
                    }
                } else {
                    // сортируем по убыванию для создания битонической последовательности
                    if(shared[tid] < shared[ixj]){
                        int temp = shared[tid];
                        shared[tid] = shared[ixj];
                        shared[ixj] = temp;
                    }
                }
            }
            
            // синхронизация после каждого шага сравнения
            // гарантирует что все потоки завершили swap перед следующим шагом
            __syncthreads();
        }
    }
    
    // запись отсортированных данных обратно в global memory
    if(idx < N){
        d_arr[idx] = shared[tid];
    }
}
// исправленная версия GPU kernel для слияния без использования shared memory
// каждый блок потоков обрабатывает одну операцию слияния
__global__ void mergeKernelNoShared(int* d_arr, int* d_temp, int N, int width){
    int mergeIdx = blockIdx.x;
    int start = 2 * width * mergeIdx;
    
    if(start >= N) return;
    
    int mid = min(start + width, N);
    int end = min(start + 2 * width, N);
    
    int leftSize = mid - start;
    int rightSize = end - mid;
    
    // только поток 0 делает слияние
    if(threadIdx.x == 0){
        int i = 0, j = 0, k = 0;
        
        while(i < leftSize && j < rightSize){
            if(d_arr[start + i] <= d_arr[mid + j]){
                d_temp[start + k++] = d_arr[start + i++];
            } else {
                d_temp[start + k++] = d_arr[mid + j++];
            }
        }
        
        while(i < leftSize){
            d_temp[start + k++] = d_arr[start + i++];
        }
        while(j < rightSize){
            d_temp[start + k++] = d_arr[mid + j++];
        }
    }
}
// GPU kernel для слияния двух отсортированных подмассивов
// каждый поток независимо сливает свою пару подмассивов
// это позволяет параллельно выполнять множество операций слияния одновременно
__global__ void mergeKernel(int* d_arr, int* d_temp, int N, int width){
    int mergeIdx = blockIdx.x;
    int start = 2 * width * mergeIdx;
    
    if(start >= N) return;
    
    int mid = min(start + width, N);
    int end = min(start + 2 * width, N);
    
    int leftSize = mid - start;
    int rightSize = end - mid;
    
    extern __shared__ int s_data[];
    
    // ПАРАЛЛЕЛЬНАЯ загрузка в shared memory
    for(int i = threadIdx.x; i < leftSize; i += blockDim.x){
        s_data[i] = d_arr[start + i];
    }
    for(int i = threadIdx.x; i < rightSize; i += blockDim.x){
        s_data[leftSize + i] = d_arr[mid + i];
    }
    __syncthreads();
    
    // ПОСЛЕДОВАТЕЛЬНОЕ слияние только потоком 0
    // это просто и гарантированно работает
    if(threadIdx.x == 0){
        int i = 0, j = 0, k = 0;
        
        // классическое слияние двух отсортированных массивов
        while(i < leftSize && j < rightSize){
            if(s_data[i] <= s_data[leftSize + j]){
                d_temp[start + k++] = s_data[i++];
            } else {
                d_temp[start + k++] = s_data[leftSize + j++];
            }
        }
        
        // копируем оставшиеся элементы
        while(i < leftSize){
            d_temp[start + k++] = s_data[i++];
        }
        while(j < rightSize){
            d_temp[start + k++] = s_data[leftSize + j++];
        }
    }
}

// главная функция GPU сортировки слиянием
void GPUMergeSort(int* arr, int N){
    int *d_arr, *d_temp;
    
    // выделение памяти на GPU
    // - cudaMalloc выделяет память в global memory GPU
    // - global memory доступна всем потокам всех блоков но имеет высокую латентность
    // - нужно два буфера: d_arr для текущих данных и d_temp для результатов слияния
    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_temp, N * sizeof(int)));
    
    // копирование данных с CPU на GPU
    // - cudaMemcpy копирует данные через PCI-E шину между host и device
    // - cudaMemcpyHostToDevice указывает направление копирования
    // - это блокирующая операция - CPU ждет завершения копирования
    CUDA_CHECK(cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice));
    
    // выбор параметров запуска kernel для сортировки блоков
    // - размер блока должен быть кратен размеру warp-а (32 потока)
    int blockSize = 256;
    
    // вычисление количества блоков для покрытия всего массива
    // формула (N + blockSize - 1) / blockSize это округление вверх
    // нужно достаточно блоков чтобы каждый элемент массива был обработан хотя бы одним потоком
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // ЭТАП 1: параллельная сортировка независимых блоков
    // - GPU состоит из множества Streaming Multiprocessors
    // - каждый SM может выполнять несколько блоков параллельно
    // - блоки полностью независимы друг от друга - нет синхронизации между блоками во время kernel
    // - это позволяет GPU эффективно распределить работу между всеми SM
    
    // запуск kernel с тремя параметрами <<<numBlocks, blockSize, sharedMemSize>>>
    // - numBlocks: количество блоков в grid
    // - blockSize: количество потоков в блоке
    // - blockSize * sizeof(int): размер динамической shared memory в байтах для каждого блока
    //   каждому блоку нужно blockSize * 4 байта для хранения своей части массива
    blockSortKernel<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_arr, N, blockSize);
    
    // проверка ошибок запуска kernel
    // - cudaGetLastError() возвращает последнюю синхронную ошибку
    // - это не гарантирует что kernel выполнился успешно только что он корректно запущен
    CUDA_CHECK(cudaGetLastError());
    
    // ожидание завершения kernel
    // - kernel запускаются асинхронно - CPU продолжает выполнение не дожидаясь завершения kernel
    // - cudaDeviceSynchronize() блокирует CPU пока все запущенные kernel не завершатся
    // - это необходимо перед следующим этапом так как слияние зависит от результатов сортировки блоков
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // ЭТАП 2: иерархическое параллельное слияние отсортированных блоков
    // - начинаем с блоков размером blockSize (уже отсортированы на этапе 1)
    // - на каждой итерации сливаем пары подмассивов размером width в подмассивы размером 2*width
    // - ширина удваивается на каждой итерации: 512 -> 1024 -> 2048 -> ... пока не достигнет N
    // - количество итераций = log2(N/blockSize)
    for(int width = blockSize; width < N; width *= 2){
    int numMerges = (N + 2 * width - 1) / (2 * width);
    int threadsPerMerge = 256;
    
    // выбираем kernel в зависимости от размера
    if(2 * width * (int)sizeof(int) <= 48 * 1024){
        // используем shared memory для малых слияний
        int sharedMemSize = 2 * width * (int)sizeof(int);
        mergeKernel<<<numMerges, threadsPerMerge, sharedMemSize>>>(d_arr, d_temp, N, width);
    } else {
        // используем версию без shared memory для больших слияний
        mergeKernelNoShared<<<numMerges, threadsPerMerge>>>(d_arr, d_temp, N, width);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // обмен указателей
    int* temp = d_arr;
    d_arr = d_temp;
    d_temp = temp;
}
    // копирование отсортированного массива обратно на CPU
    // обоснование: финальный результат находится в d_arr после всех swap-ов
    // cudaMemcpyDeviceToHost копирует данные с GPU на CPU
    CUDA_CHECK(cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    // освобождение памяти GPU
    // обоснование согласно CUDA Programming Guide Section 3.2.2:
    // - cudaFree освобождает память выделенную cudaMalloc
    // - важно освобождать память чтобы избежать memory leaks на GPU
    // - GPU memory ограничена и утечки могут привести к ошибкам в других программах
    CUDA_CHECK(cudaFree(d_arr));
    CUDA_CHECK(cudaFree(d_temp));
}

void runTest(int N){
    cout << "\n array size: " << N << endl;
    
    int* original = new int[N];
    int* arr_cpu = new int[N];
    int* arr_gpu = new int[N];
    
    for(int i = 0; i < N; i++){
        original[i] = rand() % 10000 + 1;
    }
    
    copyArray(original, arr_cpu, N);
    copyArray(original, arr_gpu, N);
    
    // тесты
    cout << "\n CPU MergeSort" << endl;
    auto start_cpu = high_resolution_clock::now();
    Sequential(arr_cpu, N);
    auto end_cpu = high_resolution_clock::now();
    duration<double> time_cpu = end_cpu - start_cpu;
    
    cout << "t: " << fixed << setprecision(6) << time_cpu.count() << " sec" << endl;
    cout << "sorted correctly: " << (isSorted(arr_cpu, N) ? "YES" : "NO") << endl;
    
    // тестирование GPU версии
    cout << "\n GPU CUDA MergeSort" << endl;
    auto start_gpu = high_resolution_clock::now();
    GPUMergeSort(arr_gpu, N);
    auto end_gpu = high_resolution_clock::now();
    duration<double> time_gpu = end_gpu - start_gpu;
    
    cout << "t: " << fixed << setprecision(6) << time_gpu.count() << " sec" << endl;
    cout << "sorted correctly: " << (isSorted(arr_gpu, N) ? "YES" : "NO") << endl;
    cout << endl;
    
    bool identical = true;
    for(int i = 0; i < N; i++){
        if(arr_cpu[i] != arr_gpu[i]){
            identical = false;
            break;
        }
    }
    
    cout << "\nSravnenie" << endl;
    cout << "identical?: " << (identical ? "YES" : "NO") << endl;
    cout << "speed: " << fixed << setprecision(2) << time_cpu.count() / time_gpu.count() << "x" << endl;
    
    delete[] original;
    delete[] arr_cpu;
    delete[] arr_gpu;
}

int main(){
    srand(time(0));
    cout << "seed: " << time(0) << endl;
    
    // получение информации о доступных GPU в системе
    // - cudaGetDeviceCount возвращает количество CUDA-совместимых устройств
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if(deviceCount > 0){
        // получение характеристик GPU
        // - cudaDeviceProp содержит множество параметров GPU
        // - эта информация полезна для оптимизации и отладки
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0)); // 0 это индекс первого GPU
        
        cout << "GPU: " << prop.name << endl;
        cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
        cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
        cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
        cout << "Number of SMs: " << prop.multiProcessorCount << endl;
    } else {
        return 1;
    }
    runTest(100000);
    runTest(1000000);
    
    return 0;
}