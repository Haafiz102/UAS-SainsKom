import numpy as np
from scipy.linalg import lu, solve
import random

#  Praktik Matriks dan Operasinya
def matrix_operations():
    print("Masukkan elemen matriks A (contoh: 1,2;3,4 untuk matriks 2x2):")
    A = np.array([list(map(float, row.split(','))) for row in input().split(';')])
    print("Masukkan elemen matriks B (ukuran sama dengan A):")
    B = np.array([list(map(float, row.split(','))) for row in input().split(';')])

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("A + B:\n", A + B)
    print("A - B:\n", A - B)
    print("A @ B (matrix multiplication):\n", A @ B)
    print("Transpose of A:\n", A.T)

# Teknik Komputasi (Invers Matriks dan LU Decomposition)
def matrix_inversion_lu_decomposition():
    print("Masukkan elemen matriks persegi A:")
    A = np.array([list(map(float, row.split(','))) for row in input().split(';')])
    print("Matrix A:\n", A)
    print("Inverse of A:\n", np.linalg.inv(A))
    P, L, U = lu(A)
    print("LU Decomposition:")
    print("L:\n", L)
    print("U:\n", U)

# Eliminasi Gauss
def gaussian_elimination():
    print("Masukkan elemen matriks A (koefisien persamaan linier):")
    A = np.array([list(map(float, row.split(','))) for row in input().split(';')])
    print("Masukkan vektor b (elemen hasil persamaan linier, pisahkan dengan koma):")
    b = np.array(list(map(float, input().split(','))))
    x = solve(A, b)
    print("Solution using Gaussian Elimination:\n", x)

# Iterasi (Jacobi dan Gauss-Seidel)
def iterative_methods():
    print("Masukkan elemen matriks A:")
    A = np.array([list(map(float, row.split(','))) for row in input().split(';')])
    print("Masukkan vektor b:")
    b = np.array(list(map(float, input().split(','))))
    x0 = np.zeros_like(b)

   
    for _ in range(10):
        x0 = (b - np.dot(A, x0) + np.diag(A) * x0) / np.diag(A)
    print("Solution using Jacobi Method:\n", x0)

    
    x0 = np.zeros_like(b)
    for _ in range(10):
        for i in range(len(b)):
            x0[i] = (b[i] - np.dot(A[i, :i], x0[:i]) - np.dot(A[i, i + 1:], x0[i + 1:])) / A[i, i]
    print("Solution using Gauss-Seidel Method:\n", x0)

#  Persamaan Linear Metode Terbuka
def open_methods():
    print("Masukkan fungsi dalam bentuk lambda (contoh: lambda x: x**2 - 4):")
    f = eval(input())
    print("Masukkan turunan fungsi (contoh: lambda x: 2*x):")
    df = eval(input())
    x = float(input("Masukkan nilai awal: "))
    
    # Newton-Raphson
    for _ in range(5):
        x = x - f(x) / df(x)
    print("Root using Newton-Raphson:\n", x)

#  Persamaan Linear Metode Tertutup
def closed_methods():
    print("Masukkan fungsi dalam bentuk lambda (contoh: lambda x: x**3 - x - 2):")
    f = eval(input())
    a, b = map(float, input("Masukkan interval (pisahkan dengan spasi): ").split())
    
    # Bisection Method
    for _ in range(10):
        c = (a + b) / 2
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    print("Root using Bisection Method:\n", c)

# Interpolasi
def interpolation():
    print("Masukkan titik x (pisahkan dengan koma):")
    x = list(map(float, input().split(',')))
    print("Masukkan nilai y yang sesuai (pisahkan dengan koma):")
    y = list(map(float, input().split(',')))
    degree = int(input("Masukkan derajat polinom: "))
    coeffs = np.polyfit(x, y, degree)
    print(f"Polynomial Coefficients (degree {degree}):\n", coeffs)

# (Monte Carlo, Markov)
def simulation_models():
    print("Monte Carlo Simulation for estimating Pi")
    trials = int(input("Masukkan jumlah percobaan: "))
    estimate = sum(random.random() ** 2 + random.random() ** 2 <= 1 for _ in range(trials)) / trials * 4
    print("Monte Carlo Estimate of Pi:\n", estimate)

    print("Markov Chain Simulation")
    print("Masukkan matriks transisi P (contoh: 0.7,0.3;0.4,0.6):")
    P = np.array([list(map(float, row.split(','))) for row in input().split(';')])
    print("Masukkan vektor status awal (pisahkan dengan koma):")
    state = np.array(list(map(float, input().split(','))))
    steps = int(input("Masukkan jumlah langkah: "))
    for _ in range(steps):
        state = np.dot(state, P)
    print("Markov Chain Final State:\n", state)


def main():
    while True:
        print("\n=== Kalkulator Numerik ===")
        print("1. Matrix Operations")
        print("2. Inversion and LU Decomposition")
        print("3. Gaussian Elimination")
        print("4. Iterative Methods (Jacobi & Gauss-Seidel)")
        print("5. Open Methods (Newton Raphson & Secant)")
        print("6. Closed Methods (Bisection)")
        print("7. Interpolation")
        print("8. Simulation Models (Monte Carlo & Markov)")
        print("9. Exit")
        choice = int(input("Pilih modul (1-9): "))
        
        if choice == 1:
            matrix_operations()
        elif choice == 2:
            matrix_inversion_lu_decomposition()
        elif choice == 3:
            gaussian_elimination()
        elif choice == 4:
            iterative_methods()
        elif choice == 5:
            open_methods()
        elif choice == 6:
            closed_methods()
        elif choice == 7:
            interpolation()
        elif choice == 8:
            simulation_models()
        elif choice == 9:
            print("Terima kasih!")
            break
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()
