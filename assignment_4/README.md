## ASSIGNMENT 1 TASK 4

Все описание кода в комментах кода.

Сравнение реализации методов с или без OpenMP

запуск g++ -fopenmp -O2 .\assignment_2\assingment_1_task_2.cpp -o task 

с флагом -fopenmp -o
и далее 

компиляция:
C:\Users\araka\Desktop\HetParallel\cpp_HeterogeneousParallelization> g++ -fopenmp -O2 .\assignment_4\assingment_1_task_4.cpp -o .\assignment_4\assignment_1_task_4

запуск:
C:\Users\araka\Desktop\HetParallel\cpp_HeterogeneousParallelization> .\assignment_4\assignment_1_task_4.exe
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