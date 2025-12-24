// insertion sort
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
    }
    
    // just array with random numbers
    // cout << "Initial array: ";
    // for(int i = 0; i < N; i++){
    //     cout << arr[i] << " ";
    // }
    // cout << endl;
    
    // insertion sort
    // этот код сортирует массив arr из N элементов по возрастанию, т.е. от меньшего к большему.
    // В общем код работает следующим образом: внешний цикл for(int i = 1; i < N; i++) проходит по каждому элементу массива, начиная со второго элемента (индекс 1). 
    // Текущий элемент сохраняется в переменную key, а внутренний цикл while(int j = i - 1; j >= 0 && arr[j] > key; j--) сдвигает элементы отсортированной части массива (слева от текущего элемента) вправо, пока не найдется правильная позиция для key. 
    // Когда правильная позиция найдена, key вставляется на это место.
    auto start = high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for(int i = 1; i < N; i++){
        // текущий элемент для вставки
        int key = arr[i];
        // индекс последнего элемента отсортированной части массива
        int j = i - 1;
        // сдвиг элементов отсортированной части массива вправо
        while(j >= 0 && arr[j] > key){
            arr[j + 1] = arr[j];
            j--;
        }
        // вставка текущего элемента на его правильную позицию
        arr[j + 1] = key;
    }
    auto end = high_resolution_clock::now();
    duration<double> duration = end - start;
    
    // sorted array
    // cout << "Sorted array: ";
    // for(int i = 0; i < N; i++){
    //     cout << arr[i] << " ";
    // }
    cout << "Insertion sort t: " << fixed << setprecision(6) << duration.count() << " sec" << endl;
    delete[] arr;
    return 0;
}