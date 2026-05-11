"""
===========================================================================
Завдання 21. Розв'язок змішаної задачі з РЧП параболічного типу
з граничними умовами різного роду.
===========================================================================

Теоретична основа (за книгою С. Фарлоу, Лекції 3, 5, 6):
----------------------------------------------------------------------
Розглядаються три типи граничних умов (ГУ):

1-й рід (Діріхле): задано значення температури на межі
   u(0, t) = g₁(t),  u(L, t) = g₂(t)

2-й рід (Нейман): задано тепловий потік на межі
   u_x(0, t) = q₁(t),  u_x(L, t) = q₂(t)
   (u_x = 0 відповідає ізольованій межі)

3-й рід (Робін/змішана): задано лінійну комбінацію
   u_x(0, t) = λ[u(0, t) - g₁(t)]  (конвективний теплообмін)

У цій програмі розглядаємо задачу з ГУ різного роду:
- Лівий кінець (x=0): ГУ 2-го роду (ізольований, u_x(0,t) = 0)
- Правий кінець (x=L): ГУ 1-го роду (фіксована температура, u(L,t) = 0)

Задача:
    u_t = α² · u_xx,   0 < x < L,  t > 0
    u_x(0, t) = 0      (ізольований лівий кінець)
    u(L, t) = 0         (нульова температура на правому кінці)
    u(x, 0) = φ(x)

Метод розділення змінних:
X''(x) + λX(x) = 0,  X'(0) = 0,  X(L) = 0

Власні функції: X_n(x) = cos((2n-1)πx / (2L)),  n = 1, 2, 3, ...
Власні значення: λ_n = ((2n-1)π / (2L))²

Розв'язок:
u(x, t) = Σ A_n · exp(-λ_n · α² · t) · cos((2n-1)πx / (2L))

Коефіцієнти:
A_n = (2/L) · ∫_0^L φ(x) · cos((2n-1)πx / (2L)) dx
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.gridspec import GridSpec

# ============================================================
# Параметри задачі
# ============================================================
alpha = 1.0     # Коефіцієнт температуропровідності
L = 1.0         # Довжина стрижня
N_terms = 50    # Кількість членів ряду
N_x = 300       # Кількість точок по x

# Моменти часу
time_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


# ============================================================
# Задача A: ГУ Нейман (x=0) + Діріхле (x=L)
# u_x(0,t)=0 (ізоляція), u(L,t)=0 (нуль на правому кінці)
# ============================================================
def eigenfunc_A(n, x, L):
    """
    Власна функція для задачі A:
    X_n(x) = cos((2n-1)πx / (2L))
    
    Задовольняє: X'(0)=0 (Нейман), X(L)=0 (Діріхле)
    """
    return np.cos((2*n - 1) * np.pi * x / (2*L))


def eigenval_A(n, L):
    """
    Власне значення: λ_n = ((2n-1)π / (2L))²
    """
    return ((2*n - 1) * np.pi / (2*L)) ** 2


def solve_problem_A(x_arr, t, alpha, L, phi_func, N_terms):
    """
    Розв'язок задачі A (Нейман-Діріхле):
    u(x,t) = Σ A_n · exp(-λ_n·α²·t) · cos((2n-1)πx/(2L))
    """
    if t <= 0:
        return phi_func(x_arr)
    
    u = np.zeros_like(x_arr, dtype=float)
    
    for n in range(1, N_terms + 1):
        # Обчислюємо коефіцієнт A_n
        def integrand(x, n=n):
            return phi_func(np.array([x]))[0] * eigenfunc_A(n, x, L)
        
        A_n, _ = quad(integrand, 0, L)
        A_n *= 2.0 / L
        
        # Власне значення
        lam_n = eigenval_A(n, L)
        
        # Додаємо n-й член ряду
        u += A_n * np.exp(-lam_n * alpha**2 * t) * eigenfunc_A(n, x_arr, L)
    
    return u


# ============================================================
# Задача B: ГУ Діріхле (x=0) + Нейман (x=L)
# u(0,t)=0 (нуль на лівому кінці), u_x(L,t)=0 (ізоляція)
# ============================================================
def eigenfunc_B(n, x, L):
    """
    Власна функція для задачі B:
    X_n(x) = sin((2n-1)πx / (2L))
    
    Задовольняє: X(0)=0 (Діріхле), X'(L)=0 (Нейман)
    """
    return np.sin((2*n - 1) * np.pi * x / (2*L))


def eigenval_B(n, L):
    """
    Власне значення: λ_n = ((2n-1)π / (2L))²
    (такі ж, як у задачі A)
    """
    return ((2*n - 1) * np.pi / (2*L)) ** 2


def solve_problem_B(x_arr, t, alpha, L, phi_func, N_terms):
    """
    Розв'язок задачі B (Діріхле-Нейман):
    u(x,t) = Σ A_n · exp(-λ_n·α²·t) · sin((2n-1)πx/(2L))
    """
    if t <= 0:
        return phi_func(x_arr)
    
    u = np.zeros_like(x_arr, dtype=float)
    
    for n in range(1, N_terms + 1):
        def integrand(x, n=n):
            return phi_func(np.array([x]))[0] * eigenfunc_B(n, x, L)
        
        A_n, _ = quad(integrand, 0, L)
        A_n *= 2.0 / L
        
        lam_n = eigenval_B(n, L)
        u += A_n * np.exp(-lam_n * alpha**2 * t) * eigenfunc_B(n, x_arr, L)
    
    return u


# ============================================================
# Задача C: ГУ Нейман (x=0) + Нейман (x=L)
# u_x(0,t)=0, u_x(L,t)=0 (обидва кінці ізольовані)
# ============================================================
def eigenfunc_C(n, x, L):
    """
    Власна функція для задачі C:
    X_0(x) = 1 (при n=0)
    X_n(x) = cos(nπx/L) при n = 1, 2, ...
    
    Задовольняє: X'(0)=0, X'(L)=0 (обидва Нейман)
    """
    if n == 0:
        return np.ones_like(x)
    return np.cos(n * np.pi * x / L)


def eigenval_C(n, L):
    """Власне значення: λ_n = (nπ/L)²"""
    return (n * np.pi / L) ** 2


def solve_problem_C(x_arr, t, alpha, L, phi_func, N_terms):
    """
    Розв'язок задачі C (Нейман-Нейман):
    u(x,t) = A_0/2 + Σ_{n=1}^{N} A_n · exp(-(nπα/L)²t) · cos(nπx/L)
    
    A_0/2 — стале середнє значення (закон збереження тепла)
    """
    if t <= 0:
        return phi_func(x_arr)
    
    # Обчислюємо A_0 (середнє значення)
    def integrand_0(x):
        return phi_func(np.array([x]))[0]
    A_0, _ = quad(integrand_0, 0, L)
    A_0 *= 2.0 / L
    
    u = np.full_like(x_arr, A_0 / 2.0, dtype=float)
    
    for n in range(1, N_terms + 1):
        def integrand(x, n=n):
            return phi_func(np.array([x]))[0] * eigenfunc_C(n, x, L)
        
        A_n, _ = quad(integrand, 0, L)
        A_n *= 2.0 / L
        
        lam_n = eigenval_C(n, L)
        u += A_n * np.exp(-lam_n * alpha**2 * t) * eigenfunc_C(n, x_arr, L)
    
    return u


# ============================================================
# Початкова умова
# ============================================================
def phi_initial(x):
    """
    Початковий розподіл: φ(x) = sin(πx) (вибрано для наочності).
    Можна замінити на будь-яку іншу функцію.
    """
    return np.sin(np.pi * x)


def phi_step(x):
    """
    Східчаста початкова умова:
    φ(x) = 1 при x < 0.5, φ(x) = 0 при x ≥ 0.5.
    """
    return np.where(x < 0.5, 1.0, 0.0)


# ============================================================
# Побудова графіків для трьох типів ГУ
# ============================================================
x = np.linspace(0, L, N_x)
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(time_values)))

problems = [
    ("A: Нейман(0) + Діріхле(L)\nu_x(0,t)=0, u(L,t)=0",
     solve_problem_A, phi_initial),
    ("B: Діріхле(0) + Нейман(L)\nu(0,t)=0, u_x(L,t)=0",
     solve_problem_B, phi_initial),
    ("C: Нейман(0) + Нейман(L)\nu_x(0,t)=0, u_x(L,t)=0\n(ізольований стрижень)",
     solve_problem_C, phi_step),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for col, (title, solver, phi_func) in enumerate(problems):
    ax = axes[col]
    
    print(f"Обчислення: {title.split(chr(10))[0]}...")
    
    for i, t in enumerate(time_values):
        u = solver(x, t, alpha, L, phi_func, N_terms)
        
        linestyle = '--' if t == 0.0 else '-'
        linewidth = 2.5 if t == 0.0 else 1.8
        
        ax.plot(x, u, color=colors[i], linestyle=linestyle,
                linewidth=linewidth, label=f't = {t}')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x, t)', fontsize=12)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

plt.suptitle(
    'Завдання 21: Змішана задача теплопровідності з ГУ різного роду\n'
    f'u_t = α²·u_xx,  α = {alpha},  L = {L}',
    fontsize=14, fontweight='bold', y=1.05
)
plt.tight_layout()
plt.savefig('graphs.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Додатковий графік: порівняння затухання для різних ГУ
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

t_array = np.linspace(0.001, 0.5, 200)

# Обчислюємо максимальну температуру для кожного типу ГУ
for title, solver, phi_func in problems:
    max_temps = []
    for t_val in t_array:
        u_t = solver(x, t_val, alpha, L, phi_func, N_terms)
        max_temps.append(np.max(u_t))
    
    label = title.split('\n')[0]
    ax2.plot(t_array, max_temps, linewidth=2, label=label)

ax2.set_xlabel('t (час)', fontsize=12)
ax2.set_ylabel('max u(x, t)', fontsize=12)
ax2.set_title('Порівняння швидкості затухання для різних ГУ', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("Завдання 21 виконано!")
print("Графіки збережено у папці task_21_mixed_parabolic/")
print("=" * 60)
