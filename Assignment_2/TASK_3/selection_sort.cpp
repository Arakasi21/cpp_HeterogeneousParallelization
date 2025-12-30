// Задача 3. Параллельная сортировка с OpenMP
// Практическое задание (25 баллов)
// Реализуйте алгоритм сортировки выбором с использованием OpenMP:
// напишите последовательную реализацию алгоритма;
// добавьте параллелизм с помощью директив OpenMP;
// проверьте производительность для массивов размером 1 000 и 10 000 элементов.
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <omp.h>
using namespace std;
using namespace std::chrono;

// пфункция которая копирует содержимое одного массива в другой и используется для подготовки данных для последовательной и параллельной реализаций
void copyArray(int* source, int* dest, int N){
    for(int i = 0; i < N; i++){
        dest[i] = source[i];
    }
}

// функция для проверки корректности сортировки, она возвращает true, если массив отсортирован по возрастанию и false в противном случае. Она просто проходит по массиву и сравнивает каждый элемент с следующим
bool isSorted(int* arr, int N){
    for(int i = 0; i < N - 1; i++){
        if(arr[i] > arr[i + 1]) return false;
    }
    return true;
}

// последовательная сортировка выбором, реализующая классический алгоритм: на каждом шаге выбирается минимальный элемент из неотсортированной части массива и меняется местами с первым элементом этой части
void Sequential(int* arr, int N){
    for(int i = 0; i < N - 1; i++){
        int min_idx = i;
        // поиск минимального элемента в неотсортированной части
        for(int j = i + 1; j < N; j++){
            if(arr[j] < arr[min_idx]){
                min_idx = j;
            }
        }
        // обмен элементов, если найденный минимум отличается от текущего элемента
        if(min_idx != i){
            // обмен arr[i] и arr[min_idx]
            int temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
        }
    }
}

// параллельная сортировка выбором
void Parallel(int* arr, int N){
    for(int i = 0; i < N - 1; i++){
        int min_idx = i;
        int min_val = arr[i];
        
        // параллельный поиск минимального элемента
        #pragma omp parallel
        {
            // локально создал чтобы избежать гонок данных, потому что несколько потоков будут искать минимум одновременно 
            int local_min_idx = min_idx;
            int local_min_val = min_val;
            // nowait используется чтобы избежать барьера после цикла for, отличается от обычного #pragma omp for тем, что не заставляет потоки ждать друг друга после выполнения цикла
            #pragma omp for nowait
            for(int j = i + 1; j < N; j++){
                // поиск локального минимума в части массива
                if(arr[j] < local_min_val){
                    local_min_val = arr[j];
                    local_min_idx = j;
                }
            }
            
            // critical директива используется для обеспечения того, чтобы только один поток мог обновлять глобальный минимум в любой момент времени, предотвращая гонки данных
            #pragma omp critical
            {
                if(local_min_val < min_val){
                    min_val = local_min_val;
                    min_idx = local_min_idx;
                }
            }
        }
        
        // обмен элементов
        if(min_idx != i){
            int temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
        }
    }
}

void runTest(int N){
    cout << "\n array size: " << N << endl;
    
    // создание и заполнение динамических массивов
    int* original = new int[N];
    int* arr_seq = new int[N];
    int* arr_par = new int[N];
    
    for(int i = 0; i < N; i++){
        // заполняем массив случайными числами от 1 до 10000
        original[i] = rand() % 10000 + 1;
    }
    
    // копируем массив для обеих реализаций
    copyArray(original, arr_seq, N);
    copyArray(original, arr_par, N);
    
    // последовательная реализация
    cout << "\nFIRST - Sequential Selection Sort" << endl;
    auto start_seq = high_resolution_clock::now();
    Sequential(arr_seq, N);
    auto end_seq = high_resolution_clock::now();
    duration<double> time_seq = end_seq - start_seq;
    
    cout << "t: " << fixed << setprecision(6) << time_seq.count() << " sec" << endl;
    cout << "sorted correctly: " << (isSorted(arr_seq, N) ? "YES" : "NO") << endl;
    
    // параллельная реализация
    cout << "\nSECOND - Parallel Selection Sort" << endl;
    auto start_par = high_resolution_clock::now();
    Parallel(arr_par, N);
    auto end_par = high_resolution_clock::now();
    duration<double> time_par = end_par - start_par;
    
    cout << "t: " << fixed << setprecision(6) << time_par.count() << " sec" << endl;
    cout << "sorted correctly: " << (isSorted(arr_par, N) ? "YES" : "NO") << endl;
    cout << endl;
    
    // проверка идентичности результатов, хотя для сортировки это не обязательно, так как может быть несколько корректных отсортированных вариантов
    bool identical = true;
    for(int i = 0; i < N; i++){
        // сравнение элементов обоих отсортированных массивов
        if(arr_seq[i] != arr_par[i]){
            identical = false;
            break;
        }
    }
    
    // сравнение результатов и вывод скорости
    cout << "\nSravnenie" << endl;
    cout << "identical?: " << (identical ? "YES" : "NO") << endl;
    cout << "speed: " << fixed << setprecision(2) << time_seq.count() / time_par.count() << "x" << endl;
    
    // освобождение памяти 
    delete[] original;
    delete[] arr_seq;
    delete[] arr_par;
}

int main(){
    srand(time(0));
    cout << "seed: " << time(0) << endl;
    cout << "threads: " << omp_get_max_threads() << endl;
    
    // тестируем для разных размеров массивов
    runTest(10000);
    runTest(100000);
    
    return 0;
}