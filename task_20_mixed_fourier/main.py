"""
Завдання 20. Змішана задача теплопровідності з ГУ 1-го роду (Діріхле).
Розв'язок методом розділення змінних (ряд Фур'є за синусами).

Задача на відрізку [0, L]:
    u_t = α² · u_xx,   0 < x < L,  t > 0
    u(0, t) = 0,  u(L, t) = 0   — граничні умови Діріхле
    u(x, 0) = φ(x)              — початкова умова

Метод розділення змінних → власні функції sin(nπx/L), λ_n = (nπ/L)².

Розв'язок:
    u(x, t) = Σ A_n · exp(−(nπα/L)²·t) · sin(nπx/L)

Коефіцієнти Фур'є:
    A_n = (2/L) · ∫₀ᴸ φ(x) · sin(nπx/L) dx
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Параметри задачі
alpha = 1.0
L = 1.0
N_terms = 50  # кількість членів ряду
x = np.linspace(0, L, 300)
time_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

# Три варіанти початкових умов φ(x)
def phi_parabolic(x):
    return x * (1 - x)  # задовольняє ГУ: φ(0)=0, φ(1)=0

def phi_sine_sum(x):
    return np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)

def phi_triangular(x):
    return np.where(x <= 0.5, 2 * x, 2 * (1 - x))

# Коефіцієнти Фур'є A_n = (2/L) · ∫₀ᴸ φ(x)·sin(nπx/L) dx
def fourier_coefficients(phi, L, N):
    coeffs = []
    for n in range(1, N + 1):
        A_n, _ = quad(lambda xi, n=n: float(phi(xi)) * np.sin(n * np.pi * xi / L), 0, L)
        coeffs.append(2.0 / L * A_n)
    return coeffs

# Розв'язок u(x, t) як ряд Фур'є
def solve(x_arr, t, coeffs):
    u = np.zeros_like(x_arr, dtype=float)
    for n, A_n in enumerate(coeffs, start=1):
        decay = np.exp(-(n * np.pi * alpha / L)**2 * t) if t > 0 else 1.0
        u += A_n * decay * np.sin(n * np.pi * x_arr / L)
    return u

# Побудова графіків
initial_conditions = [
    (phi_parabolic,  "Параболічний: φ(x) = x(1−x)"),
    (phi_sine_sum,   "Синусоїдальний: φ(x) = sin(πx) + 0.5·sin(3πx)"),
    (phi_triangular, "Трикутний: φ(x)"),
]

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(time_values)))
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for col, (phi, title) in enumerate(initial_conditions):
    ax = axes[col]
    print(f"Обчислення коефіцієнтів Фур'є: {title}...")
    coeffs = fourier_coefficients(phi, L, N_terms)

    for i, t in enumerate(time_values):
        u = solve(x, t, coeffs)
        ax.plot(x, u, color=colors[i],
                linestyle='--' if t == 0.0 else '-',
                linewidth=1.8, label=f't = {t}')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

plt.suptitle(
    'Завдання 20: Змішана задача теплопровідності (ГУ Діріхле)\n'
    'u(x,t) = Σ Aₙ · exp(−(nπα/L)²t) · sin(nπx/L)',
    fontweight='bold'
)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'graphs.png', dpi=150, bbox_inches='tight')
plt.show()

print("Завдання 20 виконано. Графік збережено: graphs.png")
