from matrix_generator import MatrixGenerator

if __name__ == "__main__":
    generator = MatrixGenerator()
    generator.generate_matrix()
    generator.save_to_file()
