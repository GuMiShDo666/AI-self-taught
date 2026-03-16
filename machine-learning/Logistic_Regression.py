import numpy as np

# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 计算损失
def compute_loss(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    
    # 防止log(0)
    eps = 1e-8
    loss = -1 / m * np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
    return loss

# 训练模型
def logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    
    for i in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        
        # 计算梯度
        dw = 1 / m * np.dot(X.T, (y_hat - y))
        db = 1 / m * np.sum(y_hat - y)
        
        w -= lr * dw
        b -= lr * db
        
        # 每100轮打印一次损失
        if i % 100 == 0:
            loss = compute_loss(X, y, w, b)
            print(f"epoch {i}, loss = {loss:.4f}")
    
    return w, b

def predict(X, w, b):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    return (y_hat >= 0.5).astype(int)

if __name__ == "__main__":
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 8],
        [8, 8]
    ], dtype=float)

    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    w, b = logistic_regression(X, y, lr=0.1, epochs=1000)

    print("\n训练完成")
    print("w =", w)
    print("b =", b)

    y_pred = predict(X, w, b)
    print("预测结果:", y_pred)
    print("真实标签:", y.astype(int))

    acc = np.mean(y_pred == y)
    print("准确率:", acc)
