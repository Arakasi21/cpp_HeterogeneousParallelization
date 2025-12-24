## ASSIGNMENT 1 TASK 4

Все описание кода в комментах кода.

Сравнение реализации методов с или без OpenMP

с флагом -fopenmp -o

компиляция:
C:\Users\araka\Desktop\HetParallel\cpp_HeterogeneousParallelization> g++ -fopenmp -O2 .\assignment_4\assingment_1_task_4.cpp -o .\assignment_4\assignment_1_task_4

запуск:
C:\Users\araka\Desktop\HetParallel\cpp_HeterogeneousParallelization> .\assignment_4\assignment_1_task_4.exe
## Результаты
```
seed: 1766586183
num of elements: 100000000

 basic
sum: 5046428213
average: 50.4643
time: 0.030988 sec

OpenMP
sum: 5046428213
average: 50.464282
time: 0.017725 sec
speed: 1.75x
```

![alt text](image.png)
