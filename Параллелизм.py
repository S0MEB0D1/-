from mpi4py import MPI
import numpy as np
import sys

# Бесконечность для обозначения отсутствия пути
INF = float('inf')

# Инициализация MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sqrt_size = int(np.sqrt(size))
if sqrt_size * sqrt_size != size or sqrt_size < 2:
    if rank == 0:
        print("Ошибка: количество процессов должно быть квадратом целого числа >= 4")
    sys.exit(1)

# Создание сеточной топологии
dims = [sqrt_size, sqrt_size]
periods = [False, False]
cart_comm = comm.Create_cart(dims, periods=periods, reorder=True)
my_coords = cart_comm.Get_coords(rank)

def input_matrix(name, is_graph=False):
    """Функция для ввода матрицы только на root-процессе"""
    if rank != 0:
        return None
    print(f"Введите размерность матрицы {name} (строки столбцы), например: 4 4")
    rows, cols = map(int, input().split())

    print(f"Введите элементы матрицы {name}, по строкам, через пробел.")
    if is_graph:
        print("Используйте 'INF' для обозначения отсутствия ребра.")

    data = []
    for i in range(rows):
        while True:
            line = input(f"Строка {i+1}: ")
            parts = line.strip().split()
            if len(parts) != cols:
                print(f"Ошибка: нужно ввести ровно {cols} значений для этой строки.")
                continue
            try:
                row = []
                for part in parts:
                    if part.upper() == 'INF':
                        row.append(INF)
                    else:
                        row.append(float(part))
                data.append(row)
                break
            except ValueError:
                print("Некорректный ввод. Используйте числа или 'INF'.")
    return np.array(data, dtype=float)


def cannon_mult(A, B, n, block_size):
    """Распараллеленное умножение матриц по алгоритму Кэнона"""
    # Разбиение матриц на блоки
    A_local = np.zeros((block_size, block_size))
    B_local = np.zeros((block_size, block_size))
    C_local = np.zeros((block_size, block_size))

    # Рассылка начальных блоков
    row_comm = cart_comm.Sub((True, False))  # Коммуникатор по строкам
    col_comm = cart_comm.Sub((False, True))  # Коммуникатор по столбцам

    row_rank = row_comm.Get_rank()
    col_rank = col_comm.Get_rank()

    # Определяем индекс текущего блока
    coords = my_coords
    row, col = coords
    offset = (row + col) % sqrt_size

    # Отправляем нужные блоки
    if rank == 0:
        A_blocks = []
        B_blocks = []
        for i in range(sqrt_size):
            for j in range(sqrt_size):
                A_block = A[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                B_block = B[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                A_blocks.append(A_block.copy())
                B_blocks.append(B_block.copy())
    else:
        A_blocks = B_blocks = None

    A_local = comm.scatter(A_blocks, root=0)
    B_local = comm.scatter(B_blocks, root=0)

    # Локальное умножение
    def safe_matmul(a, b):
        result = np.full_like(a, INF)
        for i in range(block_size):
            for k in range(block_size):
                if A_local[i, k] == INF:
                    continue
                for j in range(block_size):
                    if B_local[k, j] == INF:
                        continue
                    if result[i, j] > A_local[i, k] + B_local[k, j]:
                        result[i, j] = A_local[i, k] + B_local[k, j]
        return result

    C_local += safe_matmul(A_local, B_local)

    # Циклические сдвиги
    for s in range(1, sqrt_size):
        # Сдвигаем A влево, B вверх
        source_col, dest_col = cart_comm.Shift(1, 1)
        cart_comm.Sendrecv_replace(A_local, dest_col, recvtag=1, source=source_col, sendtag=1)

        source_row, dest_row = cart_comm.Shift(0, 1)
        cart_comm.Sendrecv_replace(B_local, dest_row, recvtag=2, source=source_row, sendtag=2)

        C_local += safe_matmul(A_local, B_local)

    return C_local


def floyd_warshall_with_path(matrix_data):
    """Флойд-Уоршел с восстановлением пути"""
    n = len(matrix_data)
    dist = matrix_data.copy()
    next = [[-1 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                next[i][j] = 0
            elif dist[i][j] != INF:
                next[i][j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next[i][j] = next[i][k]

    return dist, next


def get_path(i, j, next):
    if next[i][j] == -1:
        return None
    path = [i]
    while i != j:
        i = next[i][j]
        path.append(i)
    return path


def main():
    if rank == 0:
        print("=== Алгебраическая матричная машина с MPI и алгоритмом Кэнона ===\n")
        print("Выберите тип операции:")
        print("1 - Параллельное умножение матриц (Кэнон)")
        print("2 - Алгоритм Флойда-Уоршела с восстановлением пути")

    choice = comm.bcast(input("Ваш выбор (1 или 2): ") if rank == 0 else None, root=0)

    if choice == "1":
        if rank == 0:
            print("\nВведите первую матрицу A:")
            A = input_matrix('A')
            print("\nВведите вторую матрицу B:")
            B = input_matrix('B')
        else:
            A = None
            B = None

        A = comm.bcast(A, root=0)
        B = comm.bcast(B, root=0)

        if A is not None and B is not None:
            n = len(A)
            if len(A[0]) != len(B) or n % sqrt_size != 0:
                if rank == 0:
                    print("Ошибка: размеры матриц должны быть согласованы и делиться на sqrt(p).")
                return

            block_size = n // sqrt_size
            A_np = np.array(A)
            B_np = np.array(B)

            result_block = cannon_mult(A_np, B_np, n, block_size)

            # Сборка результата
            all_blocks = comm.gather(result_block, root=0)

            if rank == 0:
                C = np.zeros((n, n))
                idx = 0
                for i in range(sqrt_size):
                    for j in range(sqrt_size):
                        start_i = i * block_size
                        end_i = start_i + block_size
                        start_j = j * block_size
                        end_j = start_j + block_size
                        C[start_i:end_i, start_j:end_j] = all_blocks[idx]
                        idx += 1
                print("\nРезультат умножения (по алгоритму Кэнона):")
                for row in C:
                    print(['INF' if x == INF else round(x, 2) for x in row])

    elif choice == "2":
        if rank == 0:
            print("\nВведите матрицу смежности графа:")
            G = input_matrix('G', is_graph=True)
        else:
            G = None

        G = comm.bcast(G, root=0)

        if G is not None:
            n = len(G)
            dist, next = floyd_warshall_with_path(G)

            if rank == 0:
                print("\nМатрица кратчайших путей:")
                for row in dist:
                    print(['INF' if x == INF else int(x) for x in row])

                print("\nХотите узнать путь между вершинами?")
                while True:
                    ij = input("Введите начальную и конечную вершину (через пробел) или 'q' для выхода: ")
                    if ij.lower() == 'q':
                        break
                    try:
                        i, j = map(int, ij.split())
                        path = get_path(i, j, next)
                        if path:
                            print(f"Путь из {i} в {j}: {' → '.join(map(str, path))}")
                        else:
                            print(f"Нет пути из {i} в {j}")
                    except Exception as e:
                        print("Ошибка ввода. Попробуйте снова.")

    else:
        if rank == 0:
            print("Неверный выбор операции.")


if __name__ == "__main__":
    main()