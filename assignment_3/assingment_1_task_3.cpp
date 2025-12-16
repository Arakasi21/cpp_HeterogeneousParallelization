#include <iostream> // cin cout
#include <chrono> // high resolution clock
#include <ctime> // for time()
#include <cstdlib> // srand rand etc - стандартные функции
#include <iomanip> // для high resolution clock - setprecision, fixed и т.д.
#include <omp.h> // для heterogeneous parallelization, использование директив, параллельной обработки

using namespace std; // to not write std:: always
using namespace std::chrono; // to not write std::chrono always

int main(){
    // we use srand to get random seed everytime 
    srand(time(0)); // инициализируем генератор случайных чисел
    cout << "seed: " << time(0) << endl; // check the seed
    
    int N = 10000000; // юзаем сто миллионов чтобы разницу увидеть
    int* arr = new int[N]; // init dynamic array. Для выделения динамической памяти используется оператор new
    // динамическая память это куча
    // есть условная static - статическая. Всего есть стэк, статика и динамическая.
    
    cout << "num of elements: " << N << endl;
    // заполняем массив случайными числами от 1 до 100
    for(int i = 0; i < N; i++){
        arr[i] = rand()%100 + 1; // rand() возвращает число от 0 до RAND_MAX, %100 дает остаток от деления, +1 до 1-100
    }
    
    // basic without anything
    cout << "\n basic" << endl;
    int min1 = arr[0];
    int max1 = arr[0]; 
    
    // start time
    auto start1 = high_resolution_clock::now();
    // последовательный цикл
    for(int i = 0; i < N; i++){
        if(arr[i] > max1) max1 = arr[i];
        if(arr[i] < min1) min1 = arr[i];
    }
    // end time
    auto end1 = high_resolution_clock::now();
    duration<double> duration1 = end1 - start1; // вычисляем разницу между концом и началом

    cout << "time: " << fixed << setprecision(6) << duration1.count() << " sec" << endl;
    cout << "min: " << min1 << " max: " << max1 << endl;

    // с OpenMP
    cout << "\nOpenMP" << endl;
    int min2 = arr[0]; // заново инициализируем для чистоты эксперимента
    int max2 = arr[0];
    
    auto start2 = high_resolution_clock::now();
    // OpenMP директива для параллельного поиска min и max
    // reduction гарантирует что каждый поток будет работать со своей копией переменной
    // а потом результаты объединятся правильно (для min берется минимальный из всех потоков, для max максимальный)
    #pragma omp parallel for reduction(min:min2) reduction(max:max2)
    for(int i = 0; i < N; i++){
        if(arr[i] > max2) max2 = arr[i];
        if(arr[i] < min2) min2 = arr[i];
    }
    auto end2 = high_resolution_clock::now();
    duration<double> duration2 = end2 - start2;
    
    cout << "time: " << fixed << setprecision(6) << duration2.count() << " sec" << endl;
    cout << "min: " << min2 << " max: " << max2 << endl;

    // на сколько ускорилось?
    cout << "speed: " << fixed << setprecision(2) << duration1.count() / duration2.count() << "x" << endl;
    
    delete[] arr; // оператор delete [] указатель_на_динамический_массив для освобождения памяти
    // [] говорит что это массив а не один элемент
}