import numpy as np
import matplotlib.pyplot as plt

#1
np.random.seed(0)
coef_orig = 2.5
intercept_orig = 1.0
x_vals = np.linspace(0, 10, 50) #50 точок лінійно
noise = np.random.normal(0, 2, size=x_vals.shape)
y_vals = coef_orig * x_vals + intercept_orig + noise #кінецеве значення + шум

#2 мнк
def fit_linear_custom(x_data, y_data):
    x_avg = np.mean(x_data)
    y_avg = np.mean(y_data)
    num = np.sum((x_data - x_avg) * (y_data - y_avg))
    denom = np.sum((x_data - x_avg) ** 2)
    slope = num / denom #нахил
    intercept = y_avg - slope * x_avg#зсув
    return slope, intercept


#3 (оцінка полінома степеню 1 методом найменших квадратів)
slope_my, intercept_my = fit_linear_custom(x_vals, y_vals)


slope_np, intercept_np = np.polyfit(x_vals, y_vals, 1)

print(f"Реальні параметри: k = {coef_orig}, b = {intercept_orig}")
print(f"Оцінка (своя функція): k = {slope_my:.3f}, b = {intercept_my:.3f}")
print(f"Оцінка (np.polyfit): k = {slope_np:.3f}, b = {intercept_np:.3f}")

#4 графіки
plt.scatter(x_vals, y_vals, label='Шумні дані', color='gray', alpha=0.6)
plt.plot(x_vals, coef_orig * x_vals + intercept_orig, '--', label='Початкова лінія', color='black')
plt.plot(x_vals, slope_my * x_vals + intercept_my, label='МетодНайменшихКвадратів', color='blue')
plt.plot(x_vals, slope_np * x_vals + intercept_np, ':', label='polyfit', color='red')

plt.title("Завдання 1 — Метод найменших квадратів")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
