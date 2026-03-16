import numpy as np

X = np.array([
    [1, 2, 3, 4],
    [2, 1, 0, 3],
    [3, 2, 1, 1],
    [4, 0, 2, 2],
    [5, 1, 3, 0],
    [6, 2, 4, 1],
    [7, 3, 2, 2],
    [8, 2, 1, 3],
    [9, 4, 0, 1],
    [10, 3, 2, 0]
], dtype = float)

y = np.array([15.0, 10.5, 13.5, 15.0, 20.5, 26.0, 28.0, 26.5, 23.5, 27.0], dtype = float)

m, n = X.shape
X_bias = np.hstack([np.ones((m, 1)), X])
Epochs = 100000
Lr = 0.001
theta = np.zeros(n + 1, dtype = float)

for _ in range(Epochs):
    H = X_bias @ theta
    Err = H - y
    dw = (1 / m) * (X_bias.T @ Err)
    theta -= Lr * dw

mse = (1 / m) * np.sum((X_bias @ theta - y) ** 2)

print("Training Complete!")
print("Theta = ", np.round(theta, 4))
print(f"MSE = {mse:.6f}")
