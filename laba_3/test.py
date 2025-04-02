from tabulate import tabulate

if __name__ == "__main__":
    indexes = []
    for i in range(7):
        for j in range(7):
            if i <= 1 and j < 4:
                continue
            indexes.append(f"{j + 1}{i + 1}")
    matrix = [[""] * 43]
    matrix[0][41] = "V"
    matrix[0][42] = "F"
    with open("output.txt", "w") as f:
        for i in range(41):
            row = ["0"] * 43
            row[41] = "V" + indexes[i]
            row[42] = "-f" + indexes[i]
            row[i] = "A"

            # Левый сосед
            if (
                i != 0
                and i != 3
                and i != 6
                and i != 13
                and i != 20
                and i != 27
                and i != 34
            ):
                row[i - 1] = "1/h^2"
            elif i == 0 or i == 3:
                row[42] += "-mu5(y)/h^2"
            else:
                row[42] += "-mu1(y)/h^2"

            # Правый сосед
            if (
                i != 2
                and i != 5
                and i != 12
                and i != 19
                and i != 26
                and i != 33
                and i != 40
            ):
                row[i + 1] = "1/h^2"
            else:
                row[42] += "-mu2(y)/h^2"

            # Нижний сосед
            if i > 2 and i < 6:
                row[i - 3] = "1/k^2"
            elif i > 9:
                row[i - 7] = "1/k^2"
            elif i <= 2:
                row[42] += "-mu6(x)/k^2"
            else:
                row[42] += "-mu3(x)/k^2"

            # Верхний сосед
            if i < 34 and i > 2:
                row[i + 7] = "1/k^2"
            elif i <= 2:
                row[i + 3] = "1/k^2"
            else:
                row[42] += "-mu4(x)/k^2"

            matrix.append(row)
        f.write(tabulate(matrix, headers="firstrow"))
