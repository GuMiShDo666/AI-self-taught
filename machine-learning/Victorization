import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print("Victorized Version:" + str(1000 * (toc - tic)) + "ms")

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()

print("For loop:" + str(1000 * (toc - tic)) + "ms")

# Victorized Version:1.033782958984375ms
# For loop:173.44021797180176ms
