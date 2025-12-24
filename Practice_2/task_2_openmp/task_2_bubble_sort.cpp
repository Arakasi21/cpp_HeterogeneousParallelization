#include <iostream>
#include <ctime>
#include <cstdlib>
#include <iomanip> 
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;

int main(){
    srand(time(0));
    int N = 2000;
    int* arr = new int[N];
    cout << "num of elements: " << N << endl;
    int sizes[] = {1000, 10000, 100000};
    int num_threads = omp_get_max_threads();
    cout << "threads: " << num_threads << endl;
    for(int i = 0; i < N; i++){
        arr[i] = rand()%100 + 1;
        // cout << arr[i] << " ";
    }
    
    auto start = high_resolution_clock::now();
    // сортировка пузырьком, работает таким образом: внешний цикл for(int i = 0; i < N-1; i++) выполняет N-1 проход по массиву
    // Внутренний цикл for(int j = 0; j < N-i-1; j++) сравнивает соседние элементы массива. Если текущий элемент arr[j] больше следующего элемента arr[j+1], происходит обмен их значений с помощью функции swap
    // После каждого полного прохода внешнего цикла, наибольший элемент "всплывает" к концу массива, поэтому внутренний цикл сокращается на один элемент (N-i-1) с каждым проходом
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < N-1; i++){
        for(int j = 0; j < N-i-1; j++){
            if(arr[j] > arr[j+1]){
                swap(arr[j], arr[j+1]);
            }
        }
    }    
    auto end = high_resolution_clock::now();
    duration<double> duration = end - start;
    // for(int i = 0; i < N; i++){
    //     cout << arr[i] << " ";
    // }
    cout << "Bubble Sort time: " << fixed << setprecision(6) << duration.count() << " sec" << endl;
    
    delete[] arr;
    return 0;
}