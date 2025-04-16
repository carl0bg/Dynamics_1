from dataclasses import dataclass, field
from typing import List
from tabulate import tabulate


@dataclass
class NodeMatrix:
    indexes: List[str] = field(default_factory=list)
    matrix: List[List[str]] = field(default_factory=lambda: [[''] * 22])

    def generate_indexes(self):
        for i in range(3):
            for j in range(7):
                if i == 0 and j > 5:
                    continue
                self.indexes.append(f'{j + 1}{i + 1}')

    def fill_matrix(self):
        self.matrix[0][20] = 'V'
        self.matrix[0][21] = 'F'

        for i in range(20):
            row = ['0'] * 22
            row[20] = 'V' + self.indexes[i]
            row[21] = '-f' + self.indexes[i]

            if i == 5:
                row[i] = 'B'
                row[i - 1] = '2/h^2(1 + a)'
                row[i + 6] = '1/k^2'
                row[21] += '-mu3(x)/k^2 - 2/(h^2a(1 - a))'
            elif i == 12:
                row[i] = 'C'
                row[i - 1] = '1/h^2'
                row[i + 7] = '2/k^2(1 + b)'
                row[21] += '-mu2(x)/h^2 - 2/(k^2b(1 + b))'
            else:
                row[i] = 'A'
                # Левый сосед
                if i in (0, 6, 13):
                    row[21] += '-mu1(y)/h^2'
                else:
                    row[i - 1] = '1/h^2'

                # Правый сосед
                if i in (12, 19):
                    row[21] += '-mu2(y)/h^2'
                else:
                    row[i + 1] = '1/h^2'

                # Нижний сосед
                if i < 5:
                    row[21] += '-mu3(x)/k^2'
                elif 6 <= i < 13:
                    row[i - 6] = '1/k^2'
                elif i > 12:
                    row[i - 7] = '1/k^2'

                # Верхний сосед
                if i > 12:
                    row[21] += '-mu4(x)/k^2'
                elif i < 6:
                    row[i + 6] = '1/k^2'
                else:
                    row[i + 7] = '1/k^2'

            self.matrix.append(row)

    def save_to_file(self, filename: str):
        with open(filename, 'w') as f:
            f.write(tabulate(self.matrix, headers="firstrow"))


if __name__ == '__main__':
    node_matrix = NodeMatrix()
    node_matrix.generate_indexes()
    node_matrix.fill_matrix()
    node_matrix.save_to_file('result.txt')
