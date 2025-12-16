## ASSIGNMENT 1 TASK 3

Все описание кода в комментах кода.

Сравнение реализации методов с или без OpenMP

компиляция:
C:\Users\araka\Desktop\HetParallel\cpp_HeterogeneousParallelization> g++ -fopenmp -O2 .\assignment_3\assingment_1_task_3.cpp -o .\assignment_3\assignment_1_task_3

запуск:
C:\Users\araka\Desktop\HetParallel\cpp_HeterogeneousParallelization> .\assignment_3\assignment_1_task_3.exe
## Результаты
```
seed: 1765911575
num of elements: 10000000

basic без параллелизации
time: 0.002383 sec
min: 1 max: 100

OpenMP с параллелизацией
time: 0.001076 sec
min: 1 max: 100
speed: 2.21x
```
