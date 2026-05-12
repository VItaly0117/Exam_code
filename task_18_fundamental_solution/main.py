"""
Завдання 18. Фундаментальний розв'язок задачі Коші для рівняння теплопровідності.

Рівняння: u_t = α² · u_xx,   -∞ < x < ∞,  t > 0
Початкова умова: u(x, 0) = δ(x)  (одиничний тепловий імпульс)

Фундаментальний розв'язок:
    G(x, t) = 1 / √(4πα²t) · exp(-x² / (4α²t))

При t → 0: пік наростає до нескінченності (наближається до δ(x))
При t → ∞: розподіл розширюється і згасає (тепло розповсюджується)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Параметри задачі
alpha = 1.0
x = np.linspace(-5, 5, 500)
time_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

# Фундаментальний розв'язок G(x, t)
def G(x, t, alpha):
    return (1.0 / np.sqrt(4 * np.pi * alpha**2 * t)) * np.exp(-x**2 / (4 * alpha**2 * t))

# Графік 1: профілі G(x,t) для різних моментів часу
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(time_values)))

ax = axes[0]
for i, t in enumerate(time_values):
    ax.plot(x, G(x, t, alpha), color=colors[i], linewidth=2,
            label=f't = {t}  (max = {G(0, t, alpha):.3f})')

ax.set_xlabel('x')
ax.set_ylabel('G(x, t)')
ax.set_title('Фундаментальний розв\'язок G(x,t)\nдля різних моментів часу')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Графік 2: максимум G(0,t) = 1/√(4πα²t) в залежності від часу
ax2 = axes[1]
t_arr = np.linspace(0.01, 3.0, 300)
ax2.plot(t_arr, 1.0 / np.sqrt(4 * np.pi * alpha**2 * t_arr), 'r-', linewidth=2)
ax2.set_xlabel('t')
ax2.set_ylabel('G(0, t)')
ax2.set_title('Максимум G(0,t) = 1/√(4πα²t)\n(спадає як 1/√t)')
ax2.grid(True, alpha=0.3)

plt.suptitle('Завдання 18: Розповсюдження одиничного теплового імпульсу', fontweight='bold')
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'graphs.png', dpi=150, bbox_inches='tight')
plt.show()

print("Завдання 18 виконано. Графік збережено: graphs.png")
