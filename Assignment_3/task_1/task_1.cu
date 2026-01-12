#include <iostream> // cin cout
#include <cuda_runtime.h> // для CUDA функций, cudaMalloc, cudaMemcpy и т.д.
#include <chrono> // high resolution clock для замера времени
#include <iomanip> // setprecision, fixed

using namespace std;
using namespace std::chrono;

// Global memory kernel
// __global__ -  это спецификатор который говорит что функция будет выполняться на GPU
// void значит функция ничего не возвращает
// d_in - input массив в device memory (на GPU)
// d_out - output массив в device memory 
// N - размер массива
// multiplier - число на которое умножаем
__global__ void multiplyGlobalMemory(int* d_in, int* d_out, int N, int multiplier) {
    // blockIdx.x - индекс блока в grid (сетке блоков)
    // blockDim.x - размер блока (количество потоков в блоке)
    // threadIdx.x - индекс потока внутри блока
    // формула вычисляет глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы массива, чтобы не выйти за пределы
    // если idx >= N значит этот поток лишний (может быть если N не кратно blockDim.x)
    if (idx < N) {
        // каждый поток читает из глобальной памяти и пишет в глобальную память
        // глобальная память самая медленная на GPU но доступна всем потокам
        d_out[idx] = d_in[idx] * multiplier;
    }
}

// Shared memory kernel
// shared memory быстрее глобальной памяти но доступна только внутри блока
__global__ void multiplySharedMemory(int* d_in, int* d_out, int N, int multiplier) {
    // вычисляем глобальный индекс
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // __shared__ - спецификатор для разделяемой памяти
    // она выделяется на блок, а не на поток
    // extern значит размер будет указан при запуске kernel
    // динамическое выделение shared memory
    extern __shared__ int s_data[];
    
    // загружаем данные из глобальной памяти в shared memory
    // каждый поток загружает свой элемент
    if (idx < N) {
        s_data[threadIdx.x] = d_in[idx];
    }
    
    // __syncthreads() - барьер синхронизации
    // все потоки в блоке ждут пока все загрузят данные в shared memory
    __syncthreads();
    // теперь читаем из shared memory
    // умножаем и пишем результат обратно в global memory
    if (idx < N) {
        d_out[idx] = s_data[threadIdx.x] * multiplier;
    }
}

int main() {
    int N = 100000000; 
    int multiplier = 5;
    // размер в байтах для malloc. sizeof(int) обычно 4 байта
    size_t size = N * sizeof(int);
    
    cout << "n: " << N << " elements" << endl;
    cout << "x multi: " << multiplier << endl;
    cout << "memsize: " << size / 1024 / 1024 << " MB" << endl << endl;
    
    // выделяем память на host (CPU)
    // h_ префикс значит host memory
    int* h_in = new int[N]; 
    int* h_out1 = new int[N]; // global memory
    int* h_out2 = new int[N]; // shared memory
    srand(time(0));
    for (int i = 0; i < N; i++) {
        h_in[i] = rand() % 10 + 1;
    }
    // выделяем память на GPU
    // d_ префикс значит device memory
    int *d_in, *d_out;
    // cudaMalloc - выделяет память на GPU
    // передаем адрес указателя (поэтому &d_in) и размер в байтах
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    // копируем данные с host на device
    // cudaMemcpy(куда, откуда, размер, направление)
    // cudaMemcpyHostToDevice - с CPU на GPU
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    // настройка конфигурации запуска kernel
    // threadsPerBlock - количество потоков в одном блоке (обычно 256, 512 или 1024)
    // максимум 1024 потока на блок для большинства GPU
    int threadsPerBlock = 256;
    // blocksPerGrid - количество блоков в grid (сетке)
    // (N + threadsPerBlock - 1) / threadsPerBlock - округление вверх
    // например если N=1000 и threadsPerBlock=256, то нужно 4 блока (256*4=1024 >= 1000)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cout << "threadsPerBlock: " << threadsPerBlock << endl;
    cout << "blocksPerGrid: " << blocksPerGrid << endl;
    cout << "total: " << blocksPerGrid * threadsPerBlock << endl << endl;
    
    cout << " Global Memory" << endl;
    
    // обнуляем выходной массив перед первым kernel
    // cudaMemset заполняет память на GPU нулями
    cudaMemset(d_out, 0, size);
    
    auto start1 = high_resolution_clock::now();
    // запуск kernel
    // <<<blocksPerGrid, threadsPerBlock>>> - конфигурация запуска
    // первый параметр - количество блоков
    // второй параметр - количество потоков в блоке
    multiplyGlobalMemory<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N, multiplier);
    
    // проверяем ошибки после запуска kernel
    // cudaGetLastError() возвращает последнюю ошибку CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error (Global): " << cudaGetErrorString(err) << endl;
    }
    
    // cudaDeviceSynchronize() - ждем пока GPU закончит работу
    // kernel запускается асинхронно
    cudaDeviceSynchronize();
    
    auto end1 = high_resolution_clock::now();
    duration<double> duration1 = end1 - start1;
    
    // копируем результат обратно на host
    // cudaMemcpyDeviceToHost - с GPU на CPU
    cudaMemcpy(h_out1, d_out, size, cudaMemcpyDeviceToHost);
    
    cout << "t: " << fixed << setprecision(6) << duration1.count() << " sec" << endl;
    // проверяем несколько первых элементов чтобы убедиться что все работает
    cout << "first 5 elements: ";
    for (int i = 0; i < 5; i++) {
        cout << h_in[i] << "*" << multiplier << "=" << h_out1[i] << " ";
    }
    cout << endl << endl;
    
    cout << "Proverka shared memory" << endl;

    // ВАЖНО: обнуляем d_out перед вторым kernel
    // иначе может остаться старый результат
    cudaMemset(d_out, 0, size);

    // размер shared memory на блок в байтах
    // каждый блок имеет threadsPerBlock потоков
    // каждый поток работает с одним int (4 байта)
    size_t sharedMemSize = threadsPerBlock * sizeof(int);
    
    auto start2 = high_resolution_clock::now();
    
    // запуск kernel с shared memory
    // третий параметр <<<>>> - размер динамической shared memory в байтах (sharedMemSize)
    multiplySharedMemory<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N, multiplier);
    
    // проверяем ошибки после запуска kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error (Shared): " << cudaGetErrorString(err) << endl;
    }
    
    cudaDeviceSynchronize();
    
    auto end2 = high_resolution_clock::now();
    duration<double> duration2 = end2 - start2;
    // копируем результат
    cudaMemcpy(h_out2, d_out, size, cudaMemcpyDeviceToHost);
    
    cout << "t: " << fixed << setprecision(6) << duration2.count() << " sec" << endl;
    
    // проверяем результаты
    cout << "first 5 elements: ";
    for (int i = 0; i < 5; i++) {
        cout << h_in[i] << "*" << multiplier << "=" << h_out2[i] << " ";
    }
    cout << endl << endl;

    cout << "GM time: " << fixed << setprecision(6) << duration1.count() << " sec" << endl;
    cout << "SM time: " << fixed << setprecision(6) << duration2.count() << " sec" << endl;
    
    if (duration1.count() > duration2.count()) {
        cout << "xSpeed: " << fixed << setprecision(2) 
             << duration1.count() / duration2.count() << "x faster" << endl;
    } else {
        cout << "Global memory faster : " << fixed << setprecision(2) 
             << duration2.count() / duration1.count() << "x" << endl;
    }
    
    // проверяем что оба kernel дали одинаковый результат
    bool resultsMatch = true;
    for (int i = 0; i < N; i++) {
        if (h_out1[i] != h_out2[i]) {
            resultsMatch = false;
            // если нашли несовпадение - выводим где именно
            cout << "Mismatch at index " << i << ": GM=" << h_out1[i] 
                 << " vs SM=" << h_out2[i] << endl;
            break;
        }
    }
    cout << "\nis results match?: " << (resultsMatch ? "YES" : "NO") << endl;
    
    // дополнительная проверка - считаем сколько правильных результатов
    int correctCount = 0;
    for (int i = 0; i < min(100, N); i++) { // проверяем первые 100 элементов
        if (h_out1[i] == h_in[i] * multiplier) {
            correctCount++;
        }
    }
    cout << "Correctness check (first 100): " << correctCount << "/100 correct" << endl;
    
    // cudaFree - освобождает память выделенную cudaMalloc
    cudaFree(d_in);
    cudaFree(d_out);
    
    // освобождаем память на CPU
    delete[] h_in;
    delete[] h_out1;
    delete[] h_out2;
    
    return 0;
}