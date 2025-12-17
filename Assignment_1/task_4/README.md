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
seed: 1765913593
num of elements: 100000000

basic
sum: 5046603760
average: 50.466
time: 0.011862 sec

OpenMP
sum2: 5046603760
average2: 50.466038
time: 0.006890 sec
speed: 1.72x
```
