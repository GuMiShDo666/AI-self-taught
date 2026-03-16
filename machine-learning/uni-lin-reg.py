import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype = float)
y = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], dtype = float)

m = len(x)

w = 0.0
b = 0.0
lr = 0.01
epochs = 1000

for _ in range(epochs):
    H = w * x + b
    dw = (1 / m) * np.sum((H - y) * x)
    db = (1 / m) * np.sum(H - y)
    w -= lr * dw
    b -= lr * db

print("Training Complete:")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")

Mse = (1 / m) * np.sum((w * x + b - y) ** 2)
print(f"MSE = {Mse:.6f}")
