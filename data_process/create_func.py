import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

x = np.arange(-5, 5, 0.1)

sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)

# softmax（1次元・可視化用）
exp_x = np.exp(x)
softmax = exp_x / np.sum(exp_x)

plt.title('活性化関数')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

plt.plot(x, sigmoid, label="シグモイド関数")
plt.plot(x, relu, label="ReLU関数")
plt.plot(x, softmax, label="Softmax関数")

plt.legend()

# 横軸と縦軸のスケールを同じにする
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
