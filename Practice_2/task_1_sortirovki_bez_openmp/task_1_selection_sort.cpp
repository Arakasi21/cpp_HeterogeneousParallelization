// selection sort
#include <iostream>
#include <ctime>
#include <cstdlib>
using namespace std;
int main(){
    srand(time(0));
    int N = 20;
    int* arr = new int[N];
    
    for(int i = 0; i < N; i++){
        arr[i] = rand()%100 + 1;
    }
    
    // just array with random numbers
    cout << "Initial array: ";
    for(int i = 0; i < N; i++){
        cout << arr[i] << " ";
    }
    cout << endl;
    
    // selection sort
    // этот код сортирует массив arr из N элементов по возрастанию, т.е. от меньшего к большему.
    // В общем код работает следующим образом: внешний цикл for(int i = 0; i < N-1; i++) проходит по каждому элементу массива, предполагая, что текущий элемент (arr[i]) является минимальным. 
    // Внутренний цикл for(int j = i+1; j < N; j++) ищет наименьший элемент в оставшейся части массива (от i+1 до N-1). Если находится элемент меньше текущего минимума, обновляется индекс минимума (minIndex). 
    // После завершения внутреннего цикла, если найденный минимум отличается от текущего элемента, происходит обмен значений между arr[i] и arr[minIndex].
    for(int i = 0; i < N-1; i++){
        // инициализия минимального элемента как текущего
        int minIndex = i;
        // поиск минимального элемента в оставшейся части массива
        for(int j = i+1; j < N; j++){
            if(arr[j] < arr[minIndex]){
                // обновление индекса минимального элемента
                minIndex = j;
            }
        }
        // поменять местами текущий элемент с найденным минимумом
        if(minIndex != i){
            // поменять местами arr[i] и arr[minIndex]
            swap(arr[i], arr[minIndex]);
        }
    }
    
    // sorted array
    cout << "Sorted array: ";
    for(int i = 0; i < N; i++){
        cout << arr[i] << " ";
    }
    cout << endl;
    
    delete[] arr;
    return 0;
}