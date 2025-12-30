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
    // bublle sort
    auto start = high_resolution_clock::now();
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