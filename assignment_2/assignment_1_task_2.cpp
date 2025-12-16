#include <iostream> // cin cout
#include <chrono> // high resolution clock
#include <ctime> // for time()
#include <cstdlib> // srand rand etc 
#include <iomanip> // для high resolution clock - setprecision, fixed и т.д.

using namespace std; // to not write std:: always
using namespace std::chrono; // to not write std::chrono always

int main(){
    // we use srand to get random seed everytime 
    srand(time(0)); // инициализируем генератор случайных чисел
    cout << "seed: " << time(0) << endl; // check the seed
    
    int N = 1000000; // юзаем сто миллионов чтобы разницу увидеть
    int* arr = new int[N]; // init dynamic array. Для выделения динамической памяти используется оператор new
    // динамическая память это куча
    // есть условная static - статическая. Всего есть стэк, статика и динамическая.
    // В статике разные уровни доступа - вне функции, внутри функции и т.д.
    cout << "num of elements: " << N << endl;
    // заполняем массив случайными числами от 1 до 100
    for(int i = 0; i < N; i++){
        arr[i] = rand()%100 + 1; // rand() возвращает число от 0 до RAND_MAX(это 32767), % 100 дает остаток от деления, +1 сдвигает диапазон до 1-100
    }
    
    // baseline для сравнения
    int min = arr[0];
    int max = arr[0]; 
    
    // start time
    auto start1 = high_resolution_clock::now();
    // последовательный цикл
    for(int i = 0; i < N; i++){
        if(arr[i] > max) max = arr[i];
        if(arr[i] < min) min = arr[i]; 
    }
    // end time
    auto end1 = high_resolution_clock::now();
    duration<double> duration1 = end1 - start1; // вычисляем разницу между концом и началом

    cout << "t: " << fixed << setprecision(6) << duration1.count() << " sec" << endl;
    cout << "min: " << min << " max: " << max << endl;
    
    delete[] arr; // оператор delete [] указатель_на_динамический_массив для освобождения памяти
    // [] говорит что это массив а не один элемент
}