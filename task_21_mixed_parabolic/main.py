"""
Завдання 21. Змішана задача теплопровідності з ГУ різного роду.

Рівняння: u_t = α² · u_xx,   0 < x < L,  t > 0

Розглядаються три комбінації граничних умов:

Задача A: Нейман(0) + Діріхле(L)
    u_x(0,t) = 0  (ізольований лівий кінець),  u(L,t) = 0
    Власні функції: cos((2n−1)πx/(2L)),   λ_n = ((2n−1)π/(2L))²

Задача B: Діріхле(0) + Нейман(L)
    u(0,t) = 0,  u_x(L,t) = 0  (ізольований правий кінець)
    Власні функції: sin((2n−1)πx/(2L)),   λ_n = ((2n−1)π/(2L))²

Задача C: Нейман(0) + Нейман(L)
    u_x(0,t) = 0,  u_x(L,t) = 0  (обидва кінці ізольовані)
    Власні функції: 1, cos(nπx/L),   λ_n = (nπ/L)²
    Середня температура зберігається (закон збереження тепла)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Параметри
alpha = 1.0
L = 1.0
N_terms = 50
x = np.linspace(0, L, 300)
time_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

# Початкові умови
def phi_sine(x):
    return np.sin(np.pi * x)

def phi_step(x):
    return np.where(x < 0.5, 1.0, 0.0)

# Загальна функція обчислення коефіцієнта A_n
def coeff(phi, basis_fn, L):
    val, _ = quad(lambda xi: float(phi(xi)) * basis_fn(xi), 0, L)
    return 2.0 / L * val

# Задача A: Нейман(0) + Діріхле(L)
# Власні функції: X_n(x) = cos((2n-1)πx/(2L))
def solve_A(x_arr, t, phi):
    if t <= 0:
        return phi(x_arr)
    u = np.zeros_like(x_arr, dtype=float)
    for n in range(1, N_terms + 1):
        lam = ((2*n - 1) * np.pi / (2*L))**2
        X_n = np.cos((2*n - 1) * np.pi * x_arr / (2*L))
        A_n = coeff(phi, lambda xi, n=n: np.cos((2*n-1)*np.pi*xi/(2*L)), L)
        u += A_n * np.exp(-lam * alpha**2 * t) * X_n
    return u

# Задача B: Діріхле(0) + Нейман(L)
# Власні функції: X_n(x) = sin((2n-1)πx/(2L))
def solve_B(x_arr, t, phi):
    if t <= 0:
        return phi(x_arr)
    u = np.zeros_like(x_arr, dtype=float)
    for n in range(1, N_terms + 1):
        lam = ((2*n - 1) * np.pi / (2*L))**2
        X_n = np.sin((2*n - 1) * np.pi * x_arr / (2*L))
        A_n = coeff(phi, lambda xi, n=n: np.sin((2*n-1)*np.pi*xi/(2*L)), L)
        u += A_n * np.exp(-lam * alpha**2 * t) * X_n
    return u

# Задача C: Нейман(0) + Нейман(L)
# Власні функції: 1, cos(nπx/L)
def solve_C(x_arr, t, phi):
    if t <= 0:
        return phi(x_arr)
    # Вільний член (нульова мода): A_0/2, де A_0 = (2/L)·∫φ dx = середня температура · 2
    A_0 = coeff(phi, lambda xi: 1.0, L)
    u = np.full_like(x_arr, A_0 / 2.0, dtype=float)
    for n in range(1, N_terms + 1):
        lam = (n * np.pi / L)**2
        A_n = coeff(phi, lambda xi, n=n: np.cos(n*np.pi*xi/L), L)
        u += A_n * np.exp(-lam * alpha**2 * t) * np.cos(n * np.pi * x_arr / L)
    return u

# Побудова графіків
problems = [
    ("A: Нейман(0) + Діріхле(L)\nu_x(0,t)=0,  u(L,t)=0", solve_A, phi_sine),
    ("B: Діріхле(0) + Нейман(L)\nu(0,t)=0,  u_x(L,t)=0", solve_B, phi_sine),
    ("C: Нейман(0) + Нейман(L)\nu_x(0,t)=0,  u_x(L,t)=0", solve_C, phi_step),
]

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(time_values)))
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for col, (title, solver, phi) in enumerate(problems):
    ax = axes[col]
    print(f"Обчислення: {title.splitlines()[0]}...")
    for i, t in enumerate(time_values):
        u = solver(x, t, phi)
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
    'Завдання 21: Змішана задача теплопровідності з ГУ різного роду\n'
    'u_t = α²·u_xx,  α = 1,  L = 1',
    fontweight='bold'
)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'graphs.png', dpi=150, bbox_inches='tight')
plt.show()

print("Завдання 21 виконано. Графік збережено: graphs.png")
