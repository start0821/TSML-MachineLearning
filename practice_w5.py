import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

pos = np.array([[3, 7],
                [4, 6],
                [5, 6],
                [7, 7],
                [8, 5],
                [5, 5.2],
                [7, 5],
                [6, 3.75],
                [6, 4],
                [6, 5],
                [7, 5],
                [6, 4.5],
                [7, 4.5]])
neg = np.array([[4, 5],
                [5, 5],
                [6, 3],
                [7, 4],
                [9, 4],
                [5, 4],
                [5, 4.5],
                [5, 3.5],
                [7, 3.5]])

C = 0.1
# print(pos.shape[0])  => 13
X = np.ones((pos.shape[0]+neg.shape[0], 2)) # 좌표에 있는 점들을 만든다.
X[0:pos.shape[0], :] = pos
X[pos.shape[0]:pos.shape[0]+neg.shape[0], :] = neg  # X에다가 pos, neg의 값들을 넣어줌

Y = np.ones(pos.shape[0] + neg.shape[0])
Y[0:pos.shape[0]] = 1
Y[pos.shape[0]:pos.shape[0]+neg.shape[0]] = -1 # pos 좌표에는 +1 neg 좌표에는 -1을 대입해준다.

plt.figure(1, figsize = (7, 7))
plt.plot(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], 'b+', label = 'positive')
plt.plot(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0],
         X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1], 'ro', markeredgecolor = 'None', label = 'negative')
plt.legend() # legend는 표시같은 것들을 업데이트 해주는 역할임.
plt.show()
