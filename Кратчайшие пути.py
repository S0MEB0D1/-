import sys

# Бесконечность для обозначения отсутствия пути
INF = float('inf')

class Matrix:
    def __init__(self, data):
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Все строки матрицы должны иметь одинаковую длину")
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def floyd_warshall_with_path(self):
        """Применяет алгоритм Флойда-Уоршела и строит матрицу next для восстановления пути."""
        n = self.rows
        dist = [row[:] for row in self.data]
        next = [[-1 for _ in range(n)] for _ in range(n)]

        # Инициализируем next: если есть ребро i -> j, то следующая вершина — j
        for i in range(n):
            for j in range(n):
                if i == j:
                    next[i][j] = 0  # Нет пути, остаёмся на месте
                elif dist[i][j] != INF:
                    next[i][j] = j  # Прямое ребро

        # Основной цикл Флойда-Уоршела
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] != INF and dist[k][j] != INF:
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
                            next[i][j] = next[i][k]

        return Matrix(dist), next

    def __str__(self):
        def format_cell(cell):
            if cell == INF:
                return "INF"
            else:
                return str(cell)

        return '\n'.join([' '.join([format_cell(cell) for cell in row]) for row in self.data])


def input_matrix(name, is_graph=False):
    print(f"Введите размерность матрицы {name} (строки столбцы), например: 3 3")
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
                        row.append(int(part))
                data.append(row)
                break
            except ValueError:
                print("Некорректный ввод. Используйте целые числа или 'INF'.")
    return Matrix(data)


def get_path(i, j, next_matrix):
    """Восстанавливает путь от i до j, используя матрицу next."""
    if next_matrix[i][j] == -1:
        return None  # Нет пути

    path = [i]
    while i != j:
        i = next_matrix[i][j]
        path.append(i)
    return path


def main():
    print("=== Алгебраическая матричная машина с алгоритмом Флойда-Уоршела ===\n")

    print("Выберите тип операции:")
    print("1 - Сложение матриц")
    print("2 - Умножение матриц")
    print("3 - Алгоритм Флойда-Уоршела с восстановлением пути")

    choice = input("Ваш выбор (1, 2 или 3): ")

    if choice in ['1', '2']:
        print("\nВведите первую матрицу A:")
        A = input_matrix('A')
        print("\nВведите вторую матрицу B:")
        B = input_matrix('B')

        try:
            if choice == '1':
                result = A + B
                print("\nРезультат сложения A + B:")
            elif choice == '2':
                result = A * B
                print("\nРезультат умножения A * B:")

            print(result)
        except ValueError as e:
            print("Ошибка:", e)

    elif choice == '3':
        print("\nВведите матрицу смежности графа:")
        G = input_matrix('G', is_graph=True)

        if G.rows != G.cols:
            print("Ошибка: матрица должна быть квадратной для применения алгоритма Флойда-Уоршела.")
        else:
            shortest_paths, next_matrix = G.floyd_warshall_with_path()
            print("\nМатрица кратчайших путей (после применения алгоритма Флойда-Уоршела):")
            print(shortest_paths)

            print("\nХотите узнать путь между конкретными вершинами?")
            print("Введите пару вершин (от 0 до {})".format(G.rows - 1))
            while True:
                ij = input("Введите начальную и конечную вершину (через пробел) или 'q' для выхода: ")
                if ij.lower() == 'q':
                    break
                try:
                    i, j = map(int, ij.split())
                    if not (0 <= i < G.rows and 0 <= j < G.rows):
                        print("Ошибка: номер вершины вне диапазона.")
                        continue

                    path = get_path(i, j, next_matrix)
                    if path is None:
                        print(f"Нет пути из {i} в {j}")
                    else:
                        print(f"Кратчайший путь из {i} в {j}: {' → '.join(map(str, path))}")
                except Exception as e:
                    print("Ошибка ввода. Попробуйте снова.")

    else:
        print("Неверный выбор операции.")


if __name__ == "__main__":
    main()