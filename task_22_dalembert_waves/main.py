"""
Завдання 22. Рух біжучих хвиль у нескінченній струні за методом Д'Аламбера.

Задача Коші для хвильового рівняння:
    u_tt = c² · u_xx,    −∞ < x < ∞,  t > 0
    u(x, 0) = f(x)       — початкове зміщення
    u_t(x, 0) = g(x)     — початкова швидкість

Формула Д'Аламбера (1750 р.):
    u(x, t) = ½·[f(x − ct) + f(x + ct)] + 1/(2c) · ∫_{x−ct}^{x+ct} g(ξ) dξ

f(x − ct) — хвиля, що рухається вправо;
f(x + ct) — хвиля, що рухається вліво.
Початкове зміщення ділиться на дві рівні частини, які розходяться в протилежних напрямках.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Параметри
c = 1.0
x = np.linspace(-10, 10, 800)
time_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# Початкові умови (три варіанти)
def f_gauss(x):
    return np.exp(-x**2)

def f_rect(x):
    return np.where(np.abs(x) <= 1.0, 1.0, 0.0)

def f_zero(x):
    return np.zeros_like(x)

def g_zero(x):
    return np.zeros_like(x)

def g_impulse(x):
    return np.sin(x) * np.exp(-x**2 / 4)

# Розв'язок за формулою Д'Аламбера
def dalembert(x_arr, t, c, f, g):
    # Перша частина: ½·[f(x − ct) + f(x + ct)]
    u = 0.5 * (f(x_arr - c * t) + f(x_arr + c * t))
    # Друга частина: 1/(2c) · ∫_{x−ct}^{x+ct} g(ξ) dξ
    if t > 0:
        for i, xi in enumerate(x_arr):
            integral, _ = quad(lambda z: g(np.array([z]))[0], xi - c*t, xi + c*t)
            u[i] += integral / (2 * c)
    return u

# Побудова графіків
problems = [
    ("Гауссівський імпульс\nf(x) = exp(−x²),  g(x) = 0", f_gauss, g_zero),
    ("Прямокутний імпульс\nf(x) = 1 при |x|≤1,  g(x) = 0", f_rect, g_zero),
    ("Удар по струні\nf(x) = 0,  g(x) = sin(x)·exp(−x²/4)", f_zero, g_impulse),
]

colors = plt.cm.cool(np.linspace(0, 0.9, len(time_values)))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for row, (title, f, g) in enumerate(problems):
    ax = axes[row]
    print(f"Обчислення: {title.split(chr(10))[0]}...")
    for i, t in enumerate(time_values):
        u = dalembert(x, t, c, f, g)
        ls = '--' if t == 0.0 else '-'
        ax.plot(x, u, color=colors[i], linestyle=ls, linewidth=1.5, label=f't = {t}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)
    ax.legend(fontsize=9, ncol=len(time_values))
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)

plt.suptitle(
    'Завдання 22: Біжучі хвилі — метод Д\'Аламбера\n'
    'u(x,t) = ½[f(x−ct) + f(x+ct)] + 1/(2c)·∫g(ξ)dξ',
    fontweight='bold'
)
plt.tight_layout()
plt.savefig('graphs.png', dpi=150, bbox_inches='tight')
plt.show()

print("Завдання 22 виконано. Графік збережено: graphs.png")
