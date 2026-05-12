"""
Завдання 19. Розв'язок задачі Коші для рівняння теплопровідності
з тепловим імпульсом довільної форми.

Рівняння: u_t = α² · u_xx,   -∞ < x < ∞,  t > 0
Початкова умова: u(x, 0) = φ(x)  (довільна форма імпульсу)

Розв'язок через формулу згортки:
    u(x, t) = ∫ G(x - ξ, t) · φ(ξ) dξ

де G(x, t) = 1/√(4πα²t) · exp(-x²/(4α²t)) — ядро теплопровідності.

Кожна точка φ(ξ) є мікро-імпульсом, відгук системи — суперпозиція фундаментальних розв'язків.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Параметри задачі
alpha = 1.0
x = np.linspace(-8, 8, 400)
time_values = [0.0, 0.05, 0.2, 0.5, 1.0, 3.0]

# Ядро теплопровідності G(x, t)
def heat_kernel(x, t, alpha):
    return (1.0 / np.sqrt(4 * np.pi * alpha**2 * t)) * np.exp(-x**2 / (4 * alpha**2 * t))

# Три варіанти початкової умови φ(x)
def phi_rectangular(x):
    return np.where(np.abs(x) <= 1.0, 1.0, 0.0)

def phi_triangular(x):
    return np.where(np.abs(x) <= 1.0, 1.0 - np.abs(x), 0.0)

def phi_gaussian(x):
    return np.exp(-2.0 * x**2)

# Розв'язок u(x,t) через чисельне інтегрування згортки
def solve(x_arr, t, phi):
    if t <= 0:
        return phi(x_arr)
    u = np.zeros_like(x_arr, dtype=float)
    for i, xi in enumerate(x_arr):
        integrand = lambda zeta: heat_kernel(xi - zeta, t, alpha) * phi(np.array([zeta]))[0]
        u[i], _ = quad(integrand, -10, 10, limit=100)
    return u

# Побудова графіків
initial_conditions = [
    (phi_rectangular, "Прямокутний імпульс\nφ(x) = 1, |x| ≤ 1"),
    (phi_triangular,  "Трикутний імпульс\nφ(x) = 1 − |x|, |x| ≤ 1"),
    (phi_gaussian,    "Гауссівський імпульс\nφ(x) = exp(−2x²)"),
]

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(time_values)))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for col, (phi, title) in enumerate(initial_conditions):
    ax = axes[col]
    for i, t in enumerate(time_values):
        print(f"  {title.split(chr(10))[0]}, t = {t}...")
        u = solve(x, t, phi)
        ls = '--' if t == 0.0 else '-'
        ax.plot(x, u, color=colors[i], linestyle=ls, linewidth=1.8, label=f't = {t}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-8, 8)

plt.suptitle(
    'Завдання 19: Розповсюдження теплового імпульсу довільної форми\n'
    'u(x,t) = ∫ G(x−ξ, t)·φ(ξ) dξ',
    fontweight='bold'
)
plt.tight_layout()
plt.savefig('graphs.png', dpi=150, bbox_inches='tight')
plt.show()

print("Завдання 19 виконано. Графік збережено: graphs.png")
