В данной лабораторной работе выполнена реализация алгоритма подсчета произведения матриц на CPU и GPU размерностями от 100х100 до 2000х2000.

CPU : Intel Core i5 12400F , GPU - NVIDIA GeForce RTX 3070

Программа написана на Python, CUDA используется с помощью библиотеки Numba.

Парарллелизуется вычисление элементов результирующей матрицы.

График времени выполнения:
![image](https://user-images.githubusercontent.com/57503765/194962103-4f97ac5c-0fa3-495b-bec0-a1559b7f9117.png)


График ускорения :

![image](https://user-images.githubusercontent.com/57503765/194962142-40cac4e8-f7ed-4939-ba5c-c349f95b1ab6.png)
