#include <iostream> // cin cout
#include <chrono> // high resolution clock
#include <ctime> // for time()
#include <cstdlib> //srand rand etc
#include <omp.h> // heteregeneous parallelization
#include <iomanip> // для high resolution clock

using namespace std; // to not write std:: always
using namespace std::chrono;  // to not write std::chrono always

int main(){
    // we use srand to get random seed everytime 
    srand(time(0));
    cout << "Seed: " << time(0) << endl; //check the seed
    int N = 5;
    int* arr = new int[50000]; // init dynTasks: Configure Default Build TaskTasks: Configure Default Build Taskic array. Для выделения динамической памяти используется оператор new. В целом как я понял указатели используются для передачи аргумента в функцию без копирования
    // динамическая память это куча
    // есть условная static - статическая. Всего есть стэк, статика и динамическая.
    // В статике разные уровни доступа - вне функции, внутри функции и т.д.
    cout << N << endl;
    for(int i = 0; i < N; i++){
        arr[i] = rand()%100 + 1; // rand возвращает число от 0 до RAND_MAX (это 32767), % 100 дает остаток от деления (0-99), +1 сдвигает диапазон до 1-100
        cout << arr[i] << " " << endl;
        cout << arr[i] << " " << endl;
    }
    // наш "baseline", т.е. то, с чем будем сравнивать  
    int min = arr[0];
    int max = arr[0];
    // время засекаем
    auto start = high_resolution_clock::now();

    // обычное вычисление суммы всех элементов массива
    int sum = 0;
    for(int i = 0; i < N; i++){
        sum += arr[i];
    }

    cout << "sum: " << sum << endl;
    sum = sum / N;
    cout << "average: " << sum << endl;

    // засекаем время окончания
    auto end = high_resolution_clock::now();
    duration<double> duration_sec = end - start; // вычисляем разницу между концом и началом
    
    cout << "t: " << fixed << setprecision(9) << duration_sec.count() << " seconds" << endl;

    // #pragma omp parallel — создаёт параллельную секцию, где код выполняется одновременно несколькими потоками.
    // b. #pragma omp for — распределяет итерации цикла между потоками для их одновременного выполнения.
    // c. reduction — позволяет безопасно суммировать или находить максимум и минимум, используя несколько потоков.
    //#pragma omp parallel for reduction(min:min) reduction(max:max)
    //for(int i = 0; i < N; i++){
    //    if(arr[i] > max) max = arr[i];
    //    if(arr[i] < min) min = arr[i];
    //}
    cout << endl;

    cout << "min: " << min << endl;
    cout << "max: " << max << endl;

    delete[] arr; // оператор delete [] указатель_на_динамический_массив для освобождения памяти
}