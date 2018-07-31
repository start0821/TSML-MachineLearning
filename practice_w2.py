import numpy as np
import matplotlib.pyplot as plt
import csv

# %matplotlib inline

# X: Feature variable
# Y: Depedent variable
X = []
Y = []

f = open('X.csv', 'r')
csvReader = csv.reader(f)

for row in csvReader:
    X.append(row)

f = open('Y.csv', 'r')
csvReader = csv.reader(f)

for row in csvReader:
    Y.append(row)

f.close()
X = np.asarray(X, dtype = 'float64')
Y = np.asarray(Y, dtype = 'float64')

# 아래의 예제는 x의 1차항만 고려하는 선형회귀(Linear Regression) 모형입니다.

# xTemp: 13개의 Attribute 중 첫 번째 Attribute만 Feature varaible로 활용함 - xTemp[i]  =[1, x(i)]
# theta(θ): 오차의 제곱을 최소화하는 매개변수 값
# Y_est(= xTemp * θ): 위에서 구해진 tehta로 도출된 예측치

xTemp = X[:, 0:2]

theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xTemp), xTemp)), np.transpose(xTemp)), Y)

Y_est = np.dot(xTemp, theta)

# m0, c0 = argmin |Y - (m0 * xYemp + c0)|^2
# m1, c1 = argmin |Y_est - (m1 * xYemp + c1)|^2
m0, c0 = np.linalg.lstsq(xTemp, Y)[0]
m1, c1 = np.linalg.lstsq(xTemp, Y_est)[0]
newX = np.zeros((X.shape[0], 9))

newX[:, 0:2] = X[:, 0:2]

# X의 제곱항을 만들기 위하여 아래의 For-Loop를 활용함
for i in range(2, 9):
    newX[:, i] = newX[:, 1] * newX[:, i-1]

newTheta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(newX), newX)), np.transpose(newX)), Y)

newY_est = np.dot(newX, newTheta)

# m2, c2 = argmin |Y_est - (m2 * xYemp + c2)|^2
m2, c2 = np.linalg.lstsq(xTemp, newY_est)[0]
newX = np.zeros((X.shape[0], 9))

newX[:, 0:2] = X[:, 0:2]

# X의 제곱항을 만들기 위하여 아래의 For-Loop를 활용함
for i in range(2, 9):
    newX[:, i] = newX[:, 1] * newX[:, i-1]

newTheta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(newX), newX)), np.transpose(newX)), Y)

newY_est = np.dot(newX, newTheta)

# m2, c2 = argmin |Y_est - (m2 * xYemp + c2)|^2
m2, c2 = np.linalg.lstsq(xTemp, newY_est)[0]
plt.figure(1, figsize = (17, 5))

# 그래프 1
ax1 = plt.subplot(1, 3, 1)
plt.plot(X[:, 1], Y, 'ro', markeredgecolor = 'none')
plt.plot(X[:, 1], m0+c0*X[:, 1], 'r-')
plt.plot(X[:, 1], Y_est, 'bo', markeredgecolor = 'none')
plt.plot(X[:, 1], m1+c1*X[:, 1], 'b-')
plt.xlabel('Feature Variable', fontsize = 14)
plt.ylabel('Dependent Variable', fontsize = 14)

# 그래프 2
ax2 = plt.subplot(1, 3, 2, sharey = ax1)
plt.plot(X[:, 1], Y, 'ro', markeredgecolor = 'none')
plt.plot(X[:, 1], newY_est, 'go', markeredgecolor = 'none')
plt.plot(X[:, 1], m2 + c2*X[:, 1], 'g-')
plt.xlabel('Feature Variable', fontsize = 14)
plt.ylabel('Dependent Variable', fontsize = 14)

# 그래프3
ax3 = plt.subplot(1, 3, 3, sharey = ax2)
plt.plot(X[:, 1], Y, 'ro', markeredgecolor = 'none')
plt.plot(X[:, 1], Y_est, 'bo', markeredgecolor = 'none')
plt.plot(X[:, 1], newY_est, 'go', markeredgecolor = 'none')
plt.xlabel('Feature Variable', fontsize = 14)
plt.ylabel('Dependent Variable', fontsize = 14)

plt.show()
