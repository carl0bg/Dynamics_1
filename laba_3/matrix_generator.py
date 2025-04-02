from tabulate import tabulate
from typing import List
from dataclasses import dataclass, field


@dataclass
class MatrixGenerator:
    indexes: List[str] = field(default_factory=list)
    matrix: List[List[str]] = field(default_factory=lambda: [[""] * 43])

    def __post_init__(self) -> None:
        self.indexes = self._generate_indexes()
        self.matrix[0][41] = "V"
        self.matrix[0][42] = "F"

    def _generate_indexes(self) -> List[str]:
        indexes = []
        for i in range(7):
            for j in range(7):
                if i <= 1 and j < 4:
                    continue
                indexes.append(f"{j + 1}{i + 1}")
        return indexes

    def _generate_row(self, i: int) -> List[str]:
        row = ["0"] * 43
        row[41] = f"V{self.indexes[i]}"
        row[42] = f"-f{self.indexes[i]}"
        row[i] = "A"

        # Левый сосед
        if i not in {0, 3, 6, 13, 20, 27, 34}:
            row[i - 1] = "1/h^2"
        elif i in {0, 3}:
            row[42] += "-mu5(y)/h^2"
        else:
            row[42] += "-mu1(y)/h^2"

        # Правый сосед
        if i not in {2, 5, 12, 19, 26, 33, 40}:
            row[i + 1] = "1/h^2"
        else:
            row[42] += "-mu2(y)/h^2"

        # Нижний сосед
        if 2 < i < 6:
            row[i - 3] = "1/k^2"
        elif i > 9:
            row[i - 7] = "1/k^2"
        elif i <= 2:
            row[42] += "-mu6(x)/k^2"
        else:
            row[42] += "-mu3(x)/k^2"

        # Верхний сосед
        if 2 < i < 34:
            row[i + 7] = "1/k^2"
        elif i <= 2:
            row[i + 3] = "1/k^2"
        else:
            row[42] += "-mu4(x)/k^2"

        return row

    def generate_matrix(self) -> None:
        for i in range(41):
            self.matrix.append(self._generate_row(i))

    def save_to_file(self, filename: str = "result.txt") -> None:
        with open(filename, "w") as f:
            f.write(tabulate(self.matrix, headers="firstrow"))
