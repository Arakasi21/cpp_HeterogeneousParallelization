#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace std::chrono;

int main(){
    srand(time(0));
    cout << "Seed: " << time(0) << endl;
    
    int N = 100000000;
    int* arr = new int[N];
    // заполняем массив случайными числами от 1 до 1000
    for(int i = 0; i < N; i++){
        arr[i] = rand() % 100 + 1;
    }
    cout << "threads: " << omp_get_max_threads() << endl;
    // последовательная реализация 
    cout << "\n FIRST - Sequential" << endl;
    int min_seq = arr[0];
    int max_seq = arr[0];
    // замер времени
    auto start_seq = high_resolution_clock::now();
    for(int i = 0; i < N; i++){
        // поиск min и max
        if(arr[i] > max_seq) max_seq = arr[i];
        if(arr[i] < min_seq) min_seq = arr[i];
    }
    // конец замера времени
    auto end_seq = high_resolution_clock::now();
    // вычисление продолжительности
    duration<double> time_seq = end_seq - start_seq;
    
    cout << "Time: " << fixed << setprecision(6) << time_seq.count() << " sec" << endl;
    cout << "Min: " << min_seq << ", Max: " << max_seq << endl;
    
    // параллельная реализация с OpenMP
    cout << "\n SECOND - Parallel" << endl;
    int min_par = arr[0];
    int max_par = arr[0];
    
    auto start_par = high_resolution_clock::now();
    // reduction автоматически находит min/max среди всех потоков
    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for(int i = 0; i < N; i++){
        if(arr[i] > max_par) max_par = arr[i];
        if(arr[i] < min_par) min_par = arr[i];
    }
    auto end_par = high_resolution_clock::now();
    duration<double> time_par = end_par - start_par;
    
    cout << "Time: " << fixed << setprecision(6) << time_par.count() << " sec" << endl;
    cout << "Min: " << min_par << ", Max: " << max_par << endl;
    
    // сравнение
    cout << "\n=== Comparison ===" << endl;
    cout << "Speedup: " << fixed << setprecision(2) << time_seq.count() / time_par.count() << "x" << endl;
    
    delete[] arr;
    
    return 0;
}