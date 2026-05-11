"""
===========================================================================
Завдання 22. Побудова руху біжучих хвиль в нескінченній струні
за методом Даламбера.
===========================================================================

Теоретична основа (за книгою С. Фарлоу, Лекції 17-18 — "Формула Д'Аламбера"):
----------------------------------------------------------------------
Задача Коші для хвильового рівняння на нескінченній прямій:
    (РЧП)  u_tt = c² · u_xx,    -∞ < x < ∞,  t > 0
    (ПУ)   u(x, 0) = f(x)       — початкове зміщення
           u_t(x, 0) = g(x)     — початкова швидкість

Формула Д'Аламбера (1750 р.):
    u(x, t) = 1/2 · [f(x - ct) + f(x + ct)] + 1/(2c) · ∫_{x-ct}^{x+ct} g(ξ) dξ

Фізична інтерпретація:
- f(x - ct): хвиля, що рухається вправо зі швидкістю c
- f(x + ct): хвиля, що рухається вліво зі швидкістю c
- Інтегральний член: внесок початкової швидкості
- Початкове зміщення розділяється на дві рівні половини,
  які рухаються в протилежних напрямках

Загальний розв'язок хвильового рівняння:
    u(x, t) = φ(x - ct) + ψ(x + ct)
    
Це означає, що будь-який розв'язок є суперпозицією двох хвиль,
що рухаються в протилежних напрямках.
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.gridspec import GridSpec

# ============================================================
# Параметри задачі
# ============================================================
c = 1.0               # Швидкість поширення хвилі
x_min, x_max = -10, 10  # Межі просторової області
N_x = 800             # Кількість точок по x

# Моменти часу для побудови графіків
time_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


# ============================================================
# Початкові умови (різні варіанти)
# ============================================================

# --- Варіант 1: Гауссівський імпульс (тільки зміщення) ---
def f1_gauss(x):
    """
    Початкове зміщення — Гауссівський імпульс:
    f(x) = exp(-x²)
    
    Фізичний зміст: плавне початкове зміщення, локалізоване біля x=0.
    Розділиться на дві Гауссові хвилі, що рухаються в протилежних напрямках.
    """
    return np.exp(-x**2)


def g1_zero(x):
    """Початкова швидкість = 0 (струна відпущена без поштовху)"""
    return np.zeros_like(x)


# --- Варіант 2: Прямокутний імпульс (тільки зміщення) ---
def f2_rect(x):
    """
    Прямокутний імпульс (як у Фарлоу, Лекція 17):
    f(x) = 1 при |x| ≤ 1, f(x) = 0 інакше.
    
    Початкове зміщення має різкі стрибки — демонструє
    розповсюдження розривних хвиль.
    """
    return np.where(np.abs(x) <= 1.0, 1.0, 0.0)


def g2_zero(x):
    """Початкова швидкість = 0"""
    return np.zeros_like(x)


# --- Варіант 3: Тільки початкова швидкість (удар по струні) ---
def f3_zero(x):
    """Початкове зміщення = 0 (струна у рівновазі)"""
    return np.zeros_like(x)


def g3_sine(x):
    """
    Початкова швидкість — синусоїдальний імпульс:
    g(x) = sin(x) · exp(-x²/4)
    
    Фізичний зміст: удар по струні (як у фортепіано),
    локалізований біля x=0.
    """
    return np.sin(x) * np.exp(-x**2 / 4)


# ============================================================
# Формула Д'Аламбера
# ============================================================
def dalembert_solution(x_arr, t, c, f_func, g_func):
    """
    Обчислює розв'язок задачі Коші за формулою Д'Аламбера:
    
    u(x, t) = 1/2 · [f(x - ct) + f(x + ct)] + 1/(2c) · ∫_{x-ct}^{x+ct} g(ξ) dξ
    
    Параметри:
    ----------
    x_arr : numpy array
        Просторові координати
    t : float
        Момент часу
    c : float
        Швидкість поширення хвилі
    f_func : callable
        Функція початкового зміщення f(x)
    g_func : callable
        Функція початкової швидкості g(x)
    
    Повертає:
    ---------
    u : numpy array
        Зміщення струни u(x, t)
    """
    # Перший доданок: 1/2 · [f(x - ct) + f(x + ct)]
    # Це суперпозиція двох хвиль, що рухаються вправо та вліво
    u_displacement = 0.5 * (f_func(x_arr - c * t) + f_func(x_arr + c * t))
    
    # Другий доданок: 1/(2c) · ∫_{x-ct}^{x+ct} g(ξ) dξ
    # Це внесок початкової швидкості
    u_velocity = np.zeros_like(x_arr, dtype=float)
    
    if t > 0:
        for i, xi in enumerate(x_arr):
            # Межі інтегрування: від (x - ct) до (x + ct)
            lower = xi - c * t
            upper = xi + c * t
            
            # Чисельне інтегрування
            result, _ = quad(lambda zeta: g_func(np.array([zeta]))[0],
                           lower, upper)
            u_velocity[i] = result / (2 * c)
    
    return u_displacement + u_velocity


# ============================================================
# Побудова графіків
# ============================================================
x = np.linspace(x_min, x_max, N_x)

# Список задач
problems = [
    ("Гауссівський імпульс\nf(x)=exp(-x²), g(x)=0", f1_gauss, g1_zero),
    ("Прямокутний імпульс\nf(x)=1 при |x|≤1, g(x)=0", f2_rect, g2_zero),
    ("Початкова швидкість\nf(x)=0, g(x)=sin(x)·exp(-x²/4)", f3_zero, g3_sine),
]

colors = plt.cm.cool(np.linspace(0, 0.9, len(time_values)))

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

for row, (title, f_func, g_func) in enumerate(problems):
    ax = axes[row]
    
    print(f"Обчислення: {title.split(chr(10))[0]}...")
    
    for i, t in enumerate(time_values):
        u = dalembert_solution(x, t, c, f_func, g_func)
        
        linestyle = '--' if t == 0.0 else '-'
        linewidth = 2.5 if t == 0.0 else 1.5
        alpha_val = 1.0 if t == 0.0 else 0.8
        
        ax.plot(x, u, color=colors[i], linestyle=linestyle,
                linewidth=linewidth, alpha=alpha_val, label=f't = {t}')
    
    ax.set_xlabel('x (положення на струні)', fontsize=12)
    ax.set_ylabel('u(x, t) (зміщення)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', ncol=len(time_values))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.axhline(y=0, color='k', linewidth=0.5)

plt.suptitle(
    'Завдання 22: Біжучі хвилі в нескінченній струні — метод Д\'Аламбера\n'
    r'$u(x,t) = \frac{1}{2}[f(x-ct) + f(x+ct)] + \frac{1}{2c}\int_{x-ct}^{x+ct} g(\xi)\,d\xi$'
    f',  c = {c}',
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('graphs.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Додатковий графік: детальна візуалізація розходження хвиль
# ============================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

# Показуємо окремо ліву та праву хвилі для Гауссівського імпульсу
t_detail = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]

for idx, t in enumerate(t_detail):
    row = idx // 3
    col = idx % 3
    ax = axes2[row, col]
    
    # Повний розв'язок
    u_total = 0.5 * (f1_gauss(x - c*t) + f1_gauss(x + c*t))
    
    # Права хвиля: 1/2 · f(x - ct) — рухається вправо
    u_right = 0.5 * f1_gauss(x - c*t)
    
    # Ліва хвиля: 1/2 · f(x + ct) — рухається вліво
    u_left = 0.5 * f1_gauss(x + c*t)
    
    ax.fill_between(x, u_right, alpha=0.3, color='red', label='→ права хвиля')
    ax.fill_between(x, u_left, alpha=0.3, color='blue', label='← ліва хвиля')
    ax.plot(x, u_total, 'k-', linewidth=2, label='сумарне зміщення')
    ax.plot(x, u_right, 'r--', linewidth=1)
    ax.plot(x, u_left, 'b--', linewidth=1)
    
    ax.set_title(f't = {t}', fontsize=12, fontweight='bold')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.suptitle(
    'Завдання 22: Розходження Гауссівських хвиль\n'
    'Початкове зміщення розділяється на дві рівні частини, що рухаються в протилежних напрямках',
    fontsize=13, fontweight='bold', y=1.03
)
plt.tight_layout()
plt.savefig('wave_separation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 3D-візуалізація: просторово-часова діаграма
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 8), subplot_kw={'projection': '3d'})

x_3d = np.linspace(-6, 6, 300)
t_3d = np.linspace(0, 4, 100)
X, T = np.meshgrid(x_3d, t_3d)

# Розв'язок Д'Аламбера для Гауссівського імпульсу (g=0):
U = 0.5 * (f1_gauss(X - c*T) + f1_gauss(X + c*T))

surf = ax3.plot_surface(X, T, U, cmap='RdBu_r', alpha=0.85,
                        edgecolor='none', antialiased=True)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('t', fontsize=12)
ax3.set_zlabel('u(x, t)', fontsize=12)
ax3.set_title(
    'Завдання 22: 3D — біжучі хвилі (Гауссівський імпульс)',
    fontsize=13, fontweight='bold'
)
fig3.colorbar(surf, ax=ax3, shrink=0.5, aspect=10, label='u(x, t)')
ax3.view_init(elev=20, azim=-50)
plt.tight_layout()
plt.savefig('graphs_3d.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("Завдання 22 виконано!")
print("Графіки збережено у папці task_22_dalembert_waves/")
print("=" * 60)
