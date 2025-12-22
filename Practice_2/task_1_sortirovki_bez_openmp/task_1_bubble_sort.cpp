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
    
    // bublle sort
    for(int i = 0; i < N-1; i++){
        for(int j = 0; j < N-i-1; j++){
            if(arr[j] > arr[j+1]){
                swap(arr[j], arr[j+1]);
            }
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