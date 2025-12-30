#include <iostream>
#include <ctime>
#include <cstdlib>
#include <iomanip> 
#include <chrono>
using namespace std;
using namespace std::chrono;
int main(){
    srand(time(0));
    int N = 2000;
    int* arr = new int[N];
    cout << "num of elements: " << N << endl;
    for(int i = 0; i < N; i++){
        arr[i] = rand()%100 + 1;
    }
    // insertion sort
    // этот код сортирует массив arr из N элементов по возрастанию, т.е. от меньшего к большему.
    // В общем код работает следующим образом: внешний цикл for(int i = 1; i < N; i++) проходит по каждому элементу массива, начиная со второго элемента (индекс 1). 
    // Текущий элемент сохраняется в переменную key, а внутренний цикл while(int j = i - 1; j >= 0 && arr[j] > key; j--) сдвигает элементы отсортированной части массива (слева от текущего элемента) вправо, пока не найдется правильная позиция для key. 
    // Когда правильная позиция найдена, key вставляется на это место.
    auto start = high_resolution_clock::now();
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
    cout << "Insertion sort time: " << fixed << setprecision(7) << duration.count() << " sec" << endl;
    
    delete[] arr;
    return 0;
}