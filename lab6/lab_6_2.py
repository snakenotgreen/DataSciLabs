import numpy as np
import matplotlib.pyplot as plt

#дані з 1 завд
np.random.seed(0)
true_k = 2.5
true_b = 1.0
x_data = np.linspace(0, 10, 50)
noise = np.random.normal(0, 2, size=x_data.shape)
y_data = true_k * x_data + true_b + noise

#1 градієнтний спуск
def run_gradient_step(x, y, rate=0.01, steps=1000):
    k_est, b_est = 0.0, 0.0
    n = len(x)
    loss_history = []

    for _ in range(steps):
        preds = k_est * x + b_est #передбачення
        delta = y - preds #різниця між real та передбачуваними
        mse = np.mean(delta ** 2) #середньоквадратична помилка
        loss_history.append(mse) #список всіх помилок
        #похідні
        grad_k = (-2 / n) * np.sum(x * delta)
        grad_b = (-2 / n) * np.sum(delta)

        k_est -= rate * grad_k
        b_est -= rate * grad_b

    return k_est, b_est, loss_history

#2 лінія регресії на загальний графік
learning_rate = 0.01
n_iter = 1000
k_fit, b_fit, losses = run_gradient_step(x_data, y_data, rate=learning_rate, steps=n_iter)

# 3 порівняння з polyfit
k_ref, b_ref = np.polyfit(x_data, y_data, 1)

print(f"Істинні значення: k = {true_k}, b = {true_b}")
print(f"Градієнтний спуск: k = {k_fit:.3f}, b = {b_fit:.3f}")
print(f"polyfit:           k = {k_ref:.3f}, b = {b_ref:.3f}")

# 4 графіки
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, label='Дані', alpha=0.6, color='gray')
plt.plot(x_data, true_k * x_data + true_b, '--', label='Основна лінія', color='black')
plt.plot(x_data, k_fit * x_data + b_fit, label='Градієнтний спуск', color='green')
plt.plot(x_data, k_ref * x_data + b_ref, ':', label='polyfit', color='blue')

plt.title("Завдання 2 — Лінійна регресія через градієнтний спуск")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Побудова графіка збіжності
plt.figure(figsize=(8, 4))
plt.plot(losses, color='purple')
plt.title("Збіжність градієнтного спуску")
plt.xlabel("Ітерації")
plt.ylabel("MSE")
plt.grid(True)
plt.tight_layout()
plt.show()
