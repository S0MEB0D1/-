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

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])


def input_matrix(name):
    print(f"Введите размерность матрицы {name} (строки столбцы), например: 2 3")
    rows, cols = map(int, input().split())

    print(f"Введите элементы матрицы {name}, по строкам, через пробел:")
    data = []
    for i in range(rows):
        while True:
            line = input(f"Строка {i + 1}: ")
            numbers = list(map(int, line.strip().split()))
            if len(numbers) == cols:
                data.append(numbers)
                break
            else:
                print(f"Ошибка: нужно ввести ровно {cols} чисел для этой строки.")
    return Matrix(data)


def main():
    print("=== Алгебраическая матричная машина ===\n")

    print("Введите первую матрицу A:")
    A = input_matrix('A')

    print("\nВведите вторую матрицу B:")
    B = input_matrix('B')

    print("\nВыберите операцию:")
    print("1 - Сложение матриц (A + B)")
    print("2 - Умножение матриц (A * B)")

    choice = input("Ваш выбор (1 или 2): ")

    try:
        if choice == '1':
            result = A + B
            print("\nРезультат сложения A + B:")
        elif choice == '2':
            result = A * B
            print("\nРезультат умножения A * B:")
        else:
            print("Неверный выбор операции.")
            return

        print(result)
    except ValueError as e:
        print("Ошибка:", e)


if __name__ == "__main__":
    main()