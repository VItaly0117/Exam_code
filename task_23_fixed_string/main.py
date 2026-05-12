"""
Завдання 23. Коливання закріпленої струни як ряд за власними функціями.

Задача:
    u_tt = α² · u_xx,   0 < x < L,  t > 0
    u(0, t) = 0,  u(L, t) = 0   — закріплені кінці
    u(x, 0) = f(x),  u_t(x, 0) = g(x)

Метод розділення змінних дає:
    Власні функції: X_n(x) = sin(nπx/L)
    Власні частоти:  ω_n = nπα/L

Розв'язок — ряд стоячих хвиль:
    u(x, t) = Σ sin(nπx/L) · [aₙ·sin(ωₙt) + bₙ·cos(ωₙt)]

Коефіцієнти:
    bₙ = (2/L) · ∫₀ᴸ f(x)·sin(nπx/L) dx   (від зміщення)
    aₙ = (2/(nπα)) · ∫₀ᴸ g(x)·sin(nπx/L) dx  (від швидкості)

Кожен член — стояча хвиля (мода). Частоти кратні основній — звідси музикальний звук струни.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Параметри задачі
alpha = 1.0
L = 1.0
N_terms = 30
x = np.linspace(0, L, 400)
time_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Три варіанти початкових умов
def f_plucked(x):
    # Відтягнута за середину (гітара): трикутний профіль
    return np.where(x <= L/2, 2*x/L, 2*(L-x)/L)

def f_sines(x):
    # Суперпозиція гармонік
    return np.sin(np.pi*x/L) + 0.5*np.sin(3*np.pi*x/L) + 0.25*np.sin(5*np.pi*x/L)

def f_zero(x):
    return np.zeros_like(x)

def g_zero(x):
    return np.zeros_like(x)

def g_impulse(x):
    # Удар по струні (фортепіано)
    return np.sin(3 * np.pi * x / L)

# Обчислення коефіцієнтів ряду
def compute_coeffs(f, g, alpha, L, N):
    a_list, b_list = [], []
    for n in range(1, N + 1):
        b_n, _ = quad(lambda x: f(np.array([x]))[0] * np.sin(n*np.pi*x/L), 0, L)
        b_n *= 2.0 / L
        a_n, _ = quad(lambda x: g(np.array([x]))[0] * np.sin(n*np.pi*x/L), 0, L)
        a_n *= 2.0 / (n * np.pi * alpha)
        a_list.append(a_n)
        b_list.append(b_n)
    return a_list, b_list

# Розв'язок u(x,t) як ряд за власними функціями
def solve(x_arr, t, alpha, L, a_list, b_list):
    u = np.zeros_like(x_arr, dtype=float)
    for n, (a_n, b_n) in enumerate(zip(a_list, b_list), start=1):
        omega_n = n * np.pi * alpha / L
        u += np.sin(n * np.pi * x_arr / L) * (a_n * np.sin(omega_n * t) + b_n * np.cos(omega_n * t))
    return u

# Побудова графіків
problems = [
    ("Відтягнута струна (гітара)\nf(x) — трикутний профіль,  g(x) = 0", f_plucked, g_zero),
    ("Суперпозиція гармонік\nf(x) = sin(πx) + 0.5sin(3πx) + 0.25sin(5πx)", f_sines, g_zero),
    ("Удар по струні (фортепіано)\nf(x) = 0,  g(x) = sin(3πx)", f_zero, g_impulse),
]

colors = plt.cm.rainbow(np.linspace(0, 0.9, len(time_values)))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for row, (title, f, g) in enumerate(problems):
    ax = axes[row]
    print(f"Обчислення коефіцієнтів: {title.split(chr(10))[0]}...")
    a_list, b_list = compute_coeffs(f, g, alpha, L, N_terms)

    for i, t in enumerate(time_values):
        if t == 0.0:
            u = f(x)
            ls = '--'
        else:
            u = solve(x, t, alpha, L, a_list, b_list)
            ls = '-'
        ax.plot(x, u, color=colors[i], linestyle=ls, linewidth=1.5, label=f't = {t}')

    ax.set_xlabel('x (положення на струні)')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=len(time_values))
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)

plt.suptitle(
    'Завдання 23: Коливання закріпленої струни — ряд за власними функціями\n'
    'u(x,t) = Σ sin(nπx/L)·[aₙ·sin(ωₙt) + bₙ·cos(ωₙt)]',
    fontweight='bold'
)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'graphs.png', dpi=150, bbox_inches='tight')
plt.show()

print("Завдання 23 виконано. Графік збережено: graphs.png")
