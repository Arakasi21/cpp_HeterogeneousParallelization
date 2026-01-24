/*
 * Лабораторная работа: Гибридные и распределённые параллельные вычисления
 * Задание 4: Распределённая программа с MPI
 * 
 * Цель: разделить массив между процессами MPI,
 * выполнить вычисления локально и собрать результаты
 * 
 * Тестируем на 2, 4 и 8 процессах
 * 
 * Компиляция: mpic++ -o task4_mpi task4_mpi.cpp
 * Запуск: mpirun -np 4 ./task4_mpi
 */

#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;


// функция для локальной обработки части массива
// возвращает сумму квадратов элементов
double processLocalData(double* data, int count) {
    double localSum = 0.0;
    for (int i = 0; i < count; i++) {
        localSum += data[i] * data[i];  
    }
    return localSum;
}

int main(int argc, char** argv) {
    // инициализация MPI
    MPI_Init(&argc, &argv);
    
    int rank;      
    int numProcs;   

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    int N = 10000000;  
    
    if (rank == 0) {
        cout << "========================================" << endl;
        cout << " Distributed calculations MPI" << endl;
        cout << "========================================" << endl << endl;
        
        cout << "Parameters:" << endl;
        cout << "  Array size: " << N << " elements" << endl;
        cout << "  Number of processes: " << numProcs << endl;
        cout << "  Elements per process: ~" << N / numProcs << endl << endl;
    }
    
    int localSize = N / numProcs;
    int remainder = N % numProcs;
    
    if (rank == numProcs - 1) {
        localSize += remainder;
    }
    
    double* localData = new double[localSize];
    
    double* allData = nullptr;

    
    if (rank == 0) {
        cout << "STAGE 1: Initializing data on master" << endl;
        
        allData = new double[N];
        srand(42); 
        
        for (int i = 0; i < N; i++) {
            allData[i] = (double)(rand() % 100) / 100.0; 
        }
        
        cout << "  first 5 elements: ";
        for (int i = 0; i < 5; i++) {
            cout << fixed << setprecision(2) << allData[i] << " ";
        }
        cout << endl << endl;
    }
    
    // синхронизация всех процессов перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);
    
    double startTime = MPI_Wtime();  // начало замера времени
    
    if (rank == 0) {
        cout << "STAGE 2: Distributing data (MPI_Scatter/Send)..." << endl;
    }
    
    if (rank == 0) {
        // master отправляет данные всем процессам
        
        // свои данные копируем напрямую
        for (int i = 0; i < N / numProcs; i++) {
            localData[i] = allData[i];
        }
        
        // отправляем данные остальным процессам
        for (int p = 1; p < numProcs; p++) {
            int sendSize = N / numProcs;
            if (p == numProcs - 1) {
                sendSize += remainder; 
            }
            int offset = p * (N / numProcs);
            
            MPI_Send(&allData[offset], sendSize, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Status status;
        MPI_Recv(localData, localSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }
    
    double scatterTime = MPI_Wtime();
    
    
    if (rank == 0) {
        cout << "STAGE 3: Local computations..." << endl;
    }
    
    // каждый процесс вычисляет сумму квадратов своей части
    double localSum = processLocalData(localData, localSize);
    
    double computeTime = MPI_Wtime();
    
    // вывод информации о каждом процессе
    cout << "  process " << rank << ": processed " << localSize 
         << " elements, local sum = " << fixed << setprecision(2) 
         << localSum << endl;
    
    
    MPI_Barrier(MPI_COMM_WORLD); 
    
    if (rank == 0) {
        cout << endl << "STAGE 4: Gathering results (MPI_Reduce)..." << endl;
    }
    
    double globalSum = 0.0;
    
    // MPI_Reduce собирает результаты со всех процессов
    // MPI_SUM - суммирует все локальные суммы
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double endTime = MPI_Wtime();
    
    // ============================================================
    // ЭТАП 5: Вывод результатов (только master)
    // ============================================================
    
    if (rank == 0) {
        cout << endl;
        cout << "========================================" << endl;
        cout << " results" << endl;
        cout << "========================================" << endl << endl;
        
        cout << "global sum of squares: " << fixed << setprecision(2) 
             << globalSum << endl << endl;
        
        // вычисляем эталонную сумму для проверки
        double checkSum = 0.0;
        for (int i = 0; i < N; i++) {
            checkSum += allData[i] * allData[i];
        }
        
        bool correct = fabs(globalSum - checkSum) < checkSum * 0.0001;
        cout << "check: " << (correct ? "[OK]" : "[FAIL]") << endl;
        cout << "  MPI result: " << fixed << setprecision(2) << globalSum << endl;
        cout << "  reference: " << fixed << setprecision(2) << checkSum << endl << endl;
        
        // execution time   
        double totalTime = endTime - startTime;
        double scatterDuration = scatterTime - startTime;
        double computeDuration = computeTime - scatterTime;
        double reduceDuration = endTime - computeTime;
        
        cout << "========================================" << endl;
        cout << " EXECUTION TIME" << endl;
        cout << "========================================" << endl << endl;
        
        cout << "+------------------+------------+" << endl;
        cout << "| Stage            | Time (s)   |" << endl;
        cout << "+------------------+------------+" << endl;
        cout << "| Scatter (distribution) | " << setw(10) << fixed << setprecision(6) 
             << scatterDuration << " |" << endl;
        cout << "| Compute (calculation)  | " << setw(10) << fixed << setprecision(6) 
             << computeDuration << " |" << endl;
        cout << "| Reduce (gathering)    | " << setw(10) << fixed << setprecision(6) 
             << reduceDuration << " |" << endl;
        cout << "+------------------+------------+" << endl;
        cout << "| TOTAL            | " << setw(10) << fixed << setprecision(6) 
             << totalTime << " |" << endl;
        cout << "+------------------+------------+" << endl << endl;
        
        // textual time distribution graph
        cout << "time distribution:" << endl;
        double maxT = totalTime;
        
        int scatterBar = (int)((scatterDuration / maxT) * 40);
        int computeBar = (int)((computeDuration / maxT) * 40);
        int reduceBar = (int)((reduceDuration / maxT) * 40);
        
        cout << "  Scatter |";
        for (int i = 0; i < scatterBar; i++) cout << "█";
        cout << " " << fixed << setprecision(4) << scatterDuration * 1000 << " ms" << endl;
        
        cout << "  Compute |";
        for (int i = 0; i < computeBar; i++) cout << "▓";
        cout << " " << fixed << setprecision(4) << computeDuration * 1000 << " ms" << endl;
        
        cout << "  Reduce  |";
        for (int i = 0; i < reduceBar; i++) cout << "░";
        cout << " " << fixed << setprecision(4) << reduceDuration * 1000 << " ms" << endl;
        cout << endl;
        
        // оценочное время для 1 процесса
        double singleProcTime = computeDuration * numProcs;  
        double speedup = singleProcTime / computeDuration;
        double efficiency = speedup / numProcs * 100;
        
        cout << "number of processes: " << numProcs << endl;
        cout << "estimated speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
        cout << "efficiency: " << fixed << setprecision(1) << efficiency << "%" << endl << endl;
        
        cout << "note: for accurate comparison, run" << endl;
        cout << "with different numbers of processes:" << endl;
        cout << "  mpirun -np 2 ./task4_mpi" << endl;
        cout << "  mpirun -np 4 ./task4_mpi" << endl;
        cout << "  mpirun -np 8 ./task4_mpi" << endl;
        
        delete[] allData;
    }
    
    delete[] localData;
    
    MPI_Finalize();
    
    return 0;
}
