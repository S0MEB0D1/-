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

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Можно складывать только объекты типа Matrix")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Для сложения матрицы должны быть одинаковой размерности")

        result_data = [
            [a + b for a, b in zip(row_self, row_other)]
            for row_self, row_other in zip(self.data, other.data)
        ]
        return Matrix(result_data)

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Умножение поддерживается только между матрицами")
        if self.cols != other.rows:
            raise ValueError("Число столбцов первой матрицы должно совпадать с числом строк второй")

        result_data = [[0] * other.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result_data[i][j] += self.data[i][k] * other.data[k][j]

        return Matrix(result_data)

    def floyd_warshall(self):
        """Применяет алгоритм Флойда-Уоршела к матрице смежности."""
        n = self.rows
        dist = [row[:] for row in self.data]  # копируем матрицу

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] != INF and dist[k][j] != INF:
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
        return Matrix(dist)

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


def main():
    print("=== Алгебраическая матричная машина с алгоритмом Флойда-Уоршела ===\n")

    print("Выберите тип операции:")
    print("1 - Сложение матриц")
    print("2 - Умножение матриц")
    print("3 - Алгоритм Флойда-Уоршела (только квадратная матрица)")

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
            shortest_paths = G.floyd_warshall()
            print("\nМатрица кратчайших путей (после применения алгоритма Флойда-Уоршела):")
            print(shortest_paths)
    else:
        print("Неверный выбор операции.")


if __name__ == "__main__":
    main()