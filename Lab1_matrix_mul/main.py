from numba import cuda
import time
import math
import numpy as np

# Размер матриц
size = 1000

# Cоздаем матрицы для умножений и результирующую матрицу
cpu_first_array = np.random.randint(0, 5, (size, size))
cpu_second_array = np.random.randint(0, 5, (size, size))
cpu_result = np.zeros((size, size), dtype=int)

# Копируем созданные матрицы на GPU
gpu_first_array = cuda.to_device(cpu_first_array)
gpu_second_array = cuda.to_device(cpu_second_array)

# Выделяем память на GPU для результирующей матрицы
gpu_result = cuda.device_array((len(cpu_first_array), len(cpu_second_array)))


# Переменожение матриц, алгоритмы на GPU и CPU отличается только аннотацией @cuda.jit
def cpu_matmul(a, b, c):
    for i in range(size):
        for j in range(size):
            result = 0
            for z in range(size):
                result += a[i, z] * b[z, j]
            c[i, j] = result


@cuda.jit
def gpu_matmul(a, b, c):
    for i in range(size):
        for j in range(size):
            result = 0
            for z in range(size):
                result += a[i, z] * b[z, j]
            c[i, j] = result


def main():
    # Количество нитей в блоке
    threads_per_block = (32, 32)

    # Количество блоков на сетку (сетка двумерная, т.к. массив двумерный)
    blocks_per_grid_x = int(math.ceil(cpu_first_array.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(cpu_second_array.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    print("Размер сетки = ", blocks_per_grid, threads_per_block)

    print("CPU старт")
    start_time = time.time()
    cpu_matmul(cpu_first_array, cpu_second_array, cpu_result)
    stop_time = time.time()
    print("CPU стоп,  потрачено  %s секунд" % (stop_time - start_time))

    print("GPU старт")
    start_time = time.time()
    gpu_matmul[blocks_per_grid, threads_per_block](gpu_first_array, gpu_second_array, gpu_result)
    stop_time = time.time()
    print("GPU стоп,  потрачено  %s секунд" % (stop_time - start_time))


if __name__ == "__main__":
    main()
