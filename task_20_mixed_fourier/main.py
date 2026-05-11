"""
===========================================================================
Завдання 20. Розв'язок змішаної задачі теплопровідності з граничними
умовами 1-го роду на заданому проміжку з використанням розкладу
функції в ряд Фур'є.
===========================================================================

Теоретична основа (за книгою С. Фарлоу, Лекція 5 — "Змінне розділення"):
----------------------------------------------------------------------
Змішана задача теплопровідності на відрізку [0, L]:
    (РЧП) u_t = α² · u_xx,   0 < x < L,  t > 0
    (ГУ)  u(0, t) = 0,  u(L, t) = 0       — ГУ 1-го роду (Діріхле)
    (ПУ)  u(x, 0) = φ(x)                  — початкова умова

Метод розділення змінних (Фур'є):
Шукаємо розв'язок у вигляді u(x,t) = X(x)·T(t).

Підставляючи в рівняння і розділяючи змінні:
    X''(x)/X(x) = T'(t)/(α²T(t)) = -λ

З граничних умов X(0)=0, X(L)=0 знаходимо власні функції та власні значення:
    X_n(x) = sin(nπx/L),   λ_n = (nπ/L)²,   n = 1, 2, 3, ...

Часова частина:
    T_n(t) = exp(-λ_n · α² · t) = exp(-(nπα/L)² · t)

Загальний розв'язок (ряд Фур'є):
    u(x, t) = Σ_{n=1}^{∞} A_n · exp(-(nπα/L)²t) · sin(nπx/L)

де коефіцієнти Фур'є:
    A_n = (2/L) · ∫_0^L φ(x) · sin(nπx/L) dx

Фізична інтерпретація:
- Кожен член ряду — "мода" коливань, що затухає з часом
- Вищі гармоніки (великі n) затухають швидше через множник exp(-(nπα/L)²t)
- З часом розв'язок наближається до першого члена ряду (найповільніша мода)
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ============================================================
# Параметри задачі
# ============================================================
alpha = 1.0    # Коефіцієнт температуропровідності
L = 1.0        # Довжина стрижня
N_terms = 50   # Кількість членів ряду Фур'є
N_x = 300      # Кількість точок по x

# Моменти часу для побудови графіків
time_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


# ============================================================
# Початкові умови (три варіанти)
# ============================================================
def phi_parabolic(x):
    """
    Параболічний початковий розподіл:
    φ(x) = x(1-x), який задовольняє ГУ: φ(0)=0, φ(1)=0.
    
    Це типовий приклад з підручника — гладка функція,
    що добре апроксимується рядом Фур'є за синусами.
    """
    return x * (1 - x)


def phi_sine_sum(x):
    """
    Комбінація синусів (вже розкладена в ряд Фур'є):
    φ(x) = sin(πx) + 0.5·sin(3πx)
    
    Коефіцієнти Фур'є відомі точно: A_1=1, A_3=0.5, решта=0.
    """
    return np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)


def phi_triangular(x):
    """
    Трикутний початковий розподіл (аналог задачі 5 з Фарлоу, Лекція 20):
    φ(x) = 2x       при 0 ≤ x ≤ 0.5
    φ(x) = 2(1-x)   при 0.5 ≤ x ≤ 1
    
    Фізичний зміст: стрижень нагрітий з піком у центрі.
    """
    return np.where(x <= 0.5, 2 * x, 2 * (1 - x))


# ============================================================
# Обчислення коефіцієнтів Фур'є
# ============================================================
def compute_fourier_coefficients(phi_func, L, N_terms):
    """
    Обчислює коефіцієнти ряду Фур'є за синусами:
    
    A_n = (2/L) · ∫_0^L φ(x) · sin(nπx/L) dx
    
    Параметри:
    ----------
    phi_func : callable
        Функція початкового розподілу φ(x)
    L : float
        Довжина відрізка
    N_terms : int
        Кількість членів ряду
    
    Повертає:
    ---------
    coeffs : list
        Коефіцієнти A_1, A_2, ..., A_N
    """
    coeffs = []
    for n in range(1, N_terms + 1):
        # Підінтегральна функція: φ(x) · sin(nπx/L)
        def integrand(x, n=n):
            return phi_func(np.array([x]))[0] * np.sin(n * np.pi * x / L)
        
        # Чисельне обчислення інтеграла
        A_n, _ = quad(integrand, 0, L)
        A_n *= 2.0 / L
        
        coeffs.append(A_n)
    
    return coeffs


# ============================================================
# Розв'язок змішаної задачі як ряд Фур'є
# ============================================================
def solve_mixed_fourier(x_arr, t, alpha, L, coefficients):
    """
    Обчислює розв'язок змішаної задачі за формулою:
    
    u(x, t) = Σ_{n=1}^{N} A_n · exp(-(nπα/L)²·t) · sin(nπx/L)
    
    Параметри:
    ----------
    x_arr : numpy array
        Просторові координати
    t : float
        Момент часу
    alpha : float
        Коефіцієнт температуропровідності
    L : float
        Довжина стрижня
    coefficients : list
        Коефіцієнти Фур'є [A_1, A_2, ..., A_N]
    
    Повертає:
    ---------
    u : numpy array
        Температура u(x, t)
    """
    u = np.zeros_like(x_arr, dtype=float)
    
    for n_idx, A_n in enumerate(coefficients):
        n = n_idx + 1  # n = 1, 2, 3, ...
        
        # Власне значення: λ_n = (nπ/L)²
        lambda_n = (n * np.pi / L) ** 2
        
        # Часовий множник: exp(-λ_n · α² · t)
        time_factor = np.exp(-lambda_n * alpha**2 * t)
        
        # Просторова власна функція: sin(nπx/L)
        spatial_mode = np.sin(n * np.pi * x_arr / L)
        
        # Додаємо n-й член ряду
        u += A_n * time_factor * spatial_mode
    
    return u


# ============================================================
# Побудова графіків
# ============================================================
x = np.linspace(0, L, N_x)

# Список початкових умов
initial_conditions = [
    (phi_parabolic, "Параболічний: φ(x) = x(1−x)"),
    (phi_sine_sum, "Синусоїдальний: φ(x) = sin(πx) + 0.5·sin(3πx)"),
    (phi_triangular, "Трикутний: φ(x)")
]

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(time_values)))

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for col, (phi_func, title) in enumerate(initial_conditions):
    ax = axes[col]
    
    # Обчислюємо коефіцієнти Фур'є
    print(f"Обчислення коефіцієнтів Фур'є для: {title}...")
    coeffs = compute_fourier_coefficients(phi_func, L, N_terms)
    
    # Виводимо перші 5 коефіцієнтів для перевірки
    print(f"  Перші 5 коефіцієнтів: {[f'{c:.4f}' for c in coeffs[:5]]}")
    
    for i, t in enumerate(time_values):
        if t == 0.0:
            # При t=0 показуємо початкову умову
            u = phi_func(x)
            linestyle = '--'
            linewidth = 2.5
        else:
            # Обчислюємо розв'язок для моменту часу t
            u = solve_mixed_fourier(x, t, alpha, L, coeffs)
            linestyle = '-'
            linewidth = 1.8
        
        ax.plot(x, u, color=colors[i], linestyle=linestyle,
                linewidth=linewidth, label=f't = {t}')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x, t)', fontsize=12)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

plt.suptitle(
    'Завдання 20: Змішана задача теплопровідності з ГУ 1-го роду (Діріхле)\n'
    r'$u(x,t) = \sum_{n=1}^{N} A_n \, e^{-(n\pi\alpha/L)^2 t} \, \sin(n\pi x/L)$'
    f',  α = {alpha},  L = {L}',
    fontsize=13, fontweight='bold', y=1.05
)
plt.tight_layout()
plt.savefig('graphs.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Додатковий графік: вплив кількості членів ряду
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

# Ліворуч: збіжність ряду для початкової умови
ax_left = axes2[0]
terms_list = [1, 3, 5, 10, 50]
colors2 = plt.cm.Set1(np.linspace(0, 1, len(terms_list)))

# Точна початкова умова
ax_left.plot(x, phi_triangular(x), 'k-', linewidth=2.5, label='φ(x) (точна)')

coeffs_tri = compute_fourier_coefficients(phi_triangular, L, max(terms_list))

for j, N in enumerate(terms_list):
    u_approx = solve_mixed_fourier(x, 0, alpha, L, coeffs_tri[:N])
    ax_left.plot(x, u_approx, color=colors2[j], linewidth=1.5,
                 linestyle='--', label=f'N = {N} членів')

ax_left.set_xlabel('x', fontsize=12)
ax_left.set_ylabel('u(x, 0)', fontsize=12)
ax_left.set_title('Збіжність ряду Фур\'є при t = 0\n(апроксимація початкової умови)', 
                   fontsize=12, fontweight='bold')
ax_left.legend(fontsize=9)
ax_left.grid(True, alpha=0.3)

# Праворуч: затухання коефіцієнтів
ax_right = axes2[1]
n_indices = np.arange(1, N_terms + 1)
ax_right.bar(n_indices, np.abs(coeffs_tri), color='steelblue', alpha=0.7)
ax_right.set_xlabel('n (номер гармоніки)', fontsize=12)
ax_right.set_ylabel('|A_n|', fontsize=12)
ax_right.set_title('Спектр коефіцієнтів Фур\'є\n(трикутний початковий розподіл)',
                   fontsize=12, fontweight='bold')
ax_right.grid(True, alpha=0.3, axis='y')

plt.suptitle('Завдання 20: Аналіз ряду Фур\'є', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fourier_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("Завдання 20 виконано!")
print("Графіки збережено у папці task_20_mixed_fourier/")
print("=" * 60)
