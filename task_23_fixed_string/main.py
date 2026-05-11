"""
===========================================================================
Завдання 23. Коливання закріпленої струни (як суми ряду по
власних функціях).
===========================================================================

Теоретична основа (за книгою С. Фарлоу, Лекція 20 — "Коливання обмеженої
струни (стоячі хвилі)"):
----------------------------------------------------------------------
Задача про коливання струни, закріпленої на обох кінцях:
    (РЧП)  u_tt = α² · u_xx,   0 < x < L,  t > 0
    (ГУ)   u(0, t) = 0,  u(L, t) = 0       — закріплені кінці
    (ПУ)   u(x, 0) = f(x),  u_t(x, 0) = g(x) — початкові умови

Метод розділення змінних:
Шукаємо розв'язок у вигляді u(x,t) = X(x)·T(t).

Рівняння для X: X'' + λX = 0,  X(0)=0,  X(L)=0
Власні функції: X_n(x) = sin(nπx/L),  n = 1, 2, 3, ...
Власні значення: λ_n = (nπ/L)²

Рівняння для T: T'' + α²λ_n·T = 0
Розв'язок: T_n(t) = a_n·sin(nπαt/L) + b_n·cos(nπαt/L)

Повний розв'язок (ряд за власними функціями):
    u(x, t) = Σ_{n=1}^{∞} sin(nπx/L) · [a_n·sin(nπαt/L) + b_n·cos(nπαt/L)]

Коефіцієнти визначаються з початкових умов:
    b_n = (2/L) · ∫_0^L f(x) · sin(nπx/L) dx
    a_n = (2/(nπα)) · ∫_0^L g(x) · sin(nπx/L) dx

Фізична інтерпретація (стоячі хвилі):
- Кожен член ряду — стояча хвиля (мода коливань)
- n-та мода має n-1 вузлів (нерухомих точок)
- Частота n-ї моди: ω_n = nπα/L (кратна основній частоті!)
- Саме кратність частот забезпечує приємне звучання струнних інструментів
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

# ============================================================
# Параметри задачі
# ============================================================
alpha = 1.0     # Швидкість поширення хвиль у струні (α = √(T/ρ))
L = 1.0         # Довжина струни
N_terms = 30    # Кількість членів ряду (власних функцій)
N_x = 400       # Кількість точок по x


# ============================================================
# Початкові умови (три варіанти)
# ============================================================

# --- Варіант 1: Відтягнута за середину струна (гітара) ---
def f1_plucked(x):
    """
    Трикутний початковий профіль (задача 5 з Лекції 20 Фарлоу):
    Струну відтягнули за середину на висоту h.
    
    f(x) = 2hx,         0 ≤ x ≤ L/2
    f(x) = 2h(L - x),   L/2 < x ≤ L
    
    (з h = 1 для наочності)
    """
    h = 1.0
    return np.where(x <= L/2, 2*h*x/L, 2*h*(L - x)/L)


def g1_zero(x):
    """Початкова швидкість = 0 (струну відпускають)"""
    return np.zeros_like(x)


# --- Варіант 2: Комбінація синусів ---
def f2_sines(x):
    """
    Початкове зміщення — суперпозиція гармонік:
    f(x) = sin(πx/L) + 0.5·sin(3πx/L) + 0.25·sin(5πx/L)
    
    Це приклад із Фарлоу (Лекція 20, зауваження 1).
    Коефіцієнти вже відомі: b_1=1, b_3=0.5, b_5=0.25.
    """
    return (np.sin(np.pi * x / L) + 
            0.5 * np.sin(3 * np.pi * x / L) + 
            0.25 * np.sin(5 * np.pi * x / L))


def g2_zero(x):
    """Початкова швидкість = 0"""
    return np.zeros_like(x)


# --- Варіант 3: Удар по струні (тільки початкова швидкість) ---
def f3_zero(x):
    """Початкове зміщення = 0 (струна в рівновазі)"""
    return np.zeros_like(x)


def g3_impulse(x):
    """
    Початкова швидкість — локалізований імпульс (удар молоточка):
    g(x) = sin(3πx/L)
    
    Задача 2 з Лекції 20 Фарлоу.
    """
    return np.sin(3 * np.pi * x / L)


# ============================================================
# Обчислення коефіцієнтів ряду
# ============================================================
def compute_coefficients(f_func, g_func, alpha, L, N_terms):
    """
    Обчислює коефіцієнти a_n (від початкової швидкості)
    та b_n (від початкового зміщення) для ряду:
    
    u(x,t) = Σ sin(nπx/L)·[a_n·sin(nπαt/L) + b_n·cos(nπαt/L)]
    
    b_n = (2/L) · ∫_0^L f(x)·sin(nπx/L) dx
    a_n = (2/(nπα)) · ∫_0^L g(x)·sin(nπx/L) dx
    """
    a_coeffs = []
    b_coeffs = []
    
    for n in range(1, N_terms + 1):
        # Коефіцієнт b_n (від початкового зміщення)
        def integrand_b(x, n=n):
            return f_func(np.array([x]))[0] * np.sin(n * np.pi * x / L)
        
        b_n, _ = quad(integrand_b, 0, L)
        b_n *= 2.0 / L
        b_coeffs.append(b_n)
        
        # Коефіцієнт a_n (від початкової швидкості)
        def integrand_a(x, n=n):
            return g_func(np.array([x]))[0] * np.sin(n * np.pi * x / L)
        
        a_n, _ = quad(integrand_a, 0, L)
        a_n *= 2.0 / (n * np.pi * alpha)
        a_coeffs.append(a_n)
    
    return a_coeffs, b_coeffs


# ============================================================
# Розв'язок — ряд за власними функціями
# ============================================================
def solve_string_vibration(x_arr, t, alpha, L, a_coeffs, b_coeffs):
    """
    Обчислює розв'язок задачі про коливання закріпленої струни:
    
    u(x, t) = Σ_{n=1}^{N} sin(nπx/L) · [a_n·sin(nπαt/L) + b_n·cos(nπαt/L)]
    
    Кожен член ряду — стояча хвиля (мода коливань):
    - sin(nπx/L) — просторова форма n-ї моди
    - a_n·sin(ω_n·t) + b_n·cos(ω_n·t) — часова залежність
    - ω_n = nπα/L — частота n-ї моди
    """
    u = np.zeros_like(x_arr, dtype=float)
    
    for n_idx in range(len(a_coeffs)):
        n = n_idx + 1
        
        # Частота n-ї моди
        omega_n = n * np.pi * alpha / L
        
        # Просторова власна функція
        X_n = np.sin(n * np.pi * x_arr / L)
        
        # Часова залежність
        T_n = a_coeffs[n_idx] * np.sin(omega_n * t) + b_coeffs[n_idx] * np.cos(omega_n * t)
        
        # Додаємо n-ту моду коливань
        u += X_n * T_n
    
    return u


# ============================================================
# Побудова графіків
# ============================================================
x = np.linspace(0, L, N_x)

# Моменти часу для побудови
time_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

problems = [
    ("Відтягнута за середину (гітара)\n"
     "f(x) — трикутний профіль, g(x) = 0", f1_plucked, g1_zero),
    ("Суперпозиція гармонік\n"
     "f(x) = sin(πx) + 0.5sin(3πx) + 0.25sin(5πx)", f2_sines, g2_zero),
    ("Удар по струні (фортепіано)\n"
     "f(x) = 0, g(x) = sin(3πx)", f3_zero, g3_impulse),
]

colors = plt.cm.rainbow(np.linspace(0, 0.9, len(time_values)))

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

for row, (title, f_func, g_func) in enumerate(problems):
    ax = axes[row]
    
    print(f"Обчислення коефіцієнтів: {title.split(chr(10))[0]}...")
    a_coeffs, b_coeffs = compute_coefficients(f_func, g_func, alpha, L, N_terms)
    
    # Виведення перших 5 коефіцієнтів
    print(f"  b_n (зміщення): {[f'{b:.4f}' for b in b_coeffs[:5]]}")
    print(f"  a_n (швидкість): {[f'{a:.4f}' for a in a_coeffs[:5]]}")
    
    for i, t in enumerate(time_values):
        if t == 0.0:
            u = f_func(x)
            linestyle = '--'
            linewidth = 2.5
        else:
            u = solve_string_vibration(x, t, alpha, L, a_coeffs, b_coeffs)
            linestyle = '-'
            linewidth = 1.5
        
        ax.plot(x, u, color=colors[i], linestyle=linestyle,
                linewidth=linewidth, label=f't = {t}')
    
    ax.set_xlabel('x (положення на струні)', fontsize=12)
    ax.set_ylabel('u(x, t) (зміщення)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=len(time_values))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)
    ax.axhline(y=0, color='k', linewidth=0.5)

plt.suptitle(
    'Завдання 23: Коливання закріпленої струни — ряд за власними функціями\n'
    r'$u(x,t) = \sum_{n=1}^{N} \sin(n\pi x/L)\,[a_n \sin(n\pi\alpha t/L) + b_n \cos(n\pi\alpha t/L)]$',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('graphs.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Додатковий графік: окремі моди коливань (стоячі хвилі)
# ============================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

# Показуємо перші 6 мод коливань для відтягнутої струни
a_pluck, b_pluck = compute_coefficients(f1_plucked, g1_zero, alpha, L, N_terms)
t_show = 0.1  # Фіксований момент часу

for n in range(1, 7):
    row = (n - 1) // 3
    col = (n - 1) % 3
    ax = axes2[row, col]
    
    # n-та мода
    omega_n = n * np.pi * alpha / L
    X_n = np.sin(n * np.pi * x / L)
    
    # Амплітуда n-ї моди
    amplitude = np.sqrt(a_pluck[n-1]**2 + b_pluck[n-1]**2) if n <= len(b_pluck) else 0
    
    # Показуємо форму моди у різні моменти часу
    for t_mode in np.linspace(0, 2*L/alpha, 8, endpoint=False):
        T_n = (a_pluck[n-1] * np.sin(omega_n * t_mode) + 
               b_pluck[n-1] * np.cos(omega_n * t_mode))
        ax.plot(x, X_n * T_n, alpha=0.5, linewidth=1)
    
    # Огинаюча
    ax.plot(x, amplitude * np.abs(X_n), 'k--', linewidth=1.5, label='огинаюча')
    ax.plot(x, -amplitude * np.abs(X_n), 'k--', linewidth=1.5)
    
    ax.set_title(f'Мода n = {n},  ω_{n} = {omega_n:.2f},  |A_{n}| = {amplitude:.4f}',
                fontsize=10, fontweight='bold')
    ax.set_xlim(0, L)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    # Позначаємо вузли (нерухомі точки)
    nodes = np.linspace(0, L, n + 1)
    ax.scatter(nodes, np.zeros_like(nodes), color='red', s=40, zorder=5)

plt.suptitle(
    'Завдання 23: Окремі моди коливань (стоячі хвилі)\n'
    'Червоні точки — вузли (нерухомі точки), пунктир — огинаюча',
    fontsize=13, fontweight='bold', y=1.03
)
plt.tight_layout()
plt.savefig('modes.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Додатковий графік: 3D-візуалізація коливань
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 8), subplot_kw={'projection': '3d'})

x_3d = np.linspace(0, L, 200)
t_3d = np.linspace(0, 3.0, 150)
X3, T3 = np.meshgrid(x_3d, t_3d)

# Обчислюємо розв'язок на сітці
U3 = np.zeros_like(X3)
a_pluck, b_pluck = compute_coefficients(f1_plucked, g1_zero, alpha, L, N_terms)

for n_idx in range(len(a_pluck)):
    n = n_idx + 1
    omega_n = n * np.pi * alpha / L
    X_n = np.sin(n * np.pi * X3 / L)
    T_n = a_pluck[n_idx] * np.sin(omega_n * T3) + b_pluck[n_idx] * np.cos(omega_n * T3)
    U3 += X_n * T_n

surf = ax3.plot_surface(X3, T3, U3, cmap='RdBu_r', alpha=0.85,
                        edgecolor='none', antialiased=True)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('t', fontsize=12)
ax3.set_zlabel('u(x, t)', fontsize=12)
ax3.set_title(
    'Завдання 23: 3D — коливання відтягнутої струни',
    fontsize=13, fontweight='bold'
)
fig3.colorbar(surf, ax=ax3, shrink=0.5, aspect=10, label='u(x, t)')
ax3.view_init(elev=20, azim=-50)
plt.tight_layout()
plt.savefig('graphs_3d.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("Завдання 23 виконано!")
print("Графіки збережено у папці task_23_fixed_string/")
print("=" * 60)
