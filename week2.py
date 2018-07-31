# Normal Distribution, Beta Distribution, Binomial Distribution
import math
import numpy as np


class NormalDistribution:
    def __init__(self,average=0,variance=1):
        self.mean = average
        self.variance = variance
        self.distribution = np.vectorize(self.distribution)

    def distribution(self,x=0):
        return (1./(self.variance*math.sqrt(math.pi*2.)))*math.exp(-(x-self.mean)**2/(2.*self.variance**2))

def N(x,average=0,variance=1):
    return (1./(variance*math.sqrt(math.pi*2.)))*math.exp(-(x-average)**2/(2.*variance**2))
N = np.vectorize(N)


class BetaDistribution:
    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        self.B = math.gamma(self.alpha)*math.gamma(self.beta)/math.gamma(self.beta+self.alpha)
        self.distribution = np.vectorize(self.distribution)
        self.mean = self.alpha/(self.alpha+self.beta)
        self.variance = self.alpha*self.beta/((self.alpha+self.beta)**2*(self.alpha+self.beta+1))

    def distribution(self,x):
        return (x**(self.alpha-1)*(1-x)**(self.beta-1))/self.B

def Beta(x,alpha,beta):
    B = math.gamma(alpha)*math.gamma(beta)/math.gamma(beta+alpha)
    return (x**alpha*(1-x)**beta)/B
Beta = np.vectorize(Beta)


class BinomialDistribution:
    def __init__(self,n,p):
        self.p = p
        self.n = n
        self.distribution = np.vectorize(self.distribution)
        self.mean = self.n*self.p
        self.variance = self.n*self.p*(1-self.p)

    def distribution(self,x):
        return math.factorial(self.n)/(math.factorial(x)*math.factorial(self.n-x)) * (self.p**x) * ((1-self.p)**(self.n-x))

def Bi(x,alpha,beta):
    return math.factorial(self.n)/(math.factorial(x)*math.factorial(self.n-x)) * (self.p**x) * ((1-self.p)**(self.n-x))
Bi = np.vectorize(Bi)


# class MultinomialDistribution:
#     def __init__(self,*args):
#         print(len(args))
#         self.n = np.array([args[i] for i in range(len(args)//2)])
#         self.p = np.array([args[i] for i in range(len(args)//2+1,len(args))])
#         self.distribution = np.vectorize(self.distribution)
#         self.mean = np.sum(self.n*self.p)
#         self.variance = np.sum(self.n*self.p*(1-self.p))
#
#     def distribution(self,x):
#         return math.factorial(self.n)/(math.factorial(x)*math.factorial(self.n-x)) * (self.p**x) * ((1-self.p)**(self.n-x))
#
# def Multi(*args):
#     n = np.array([args[i] for i in range(len(args)//2)])
#     p = np.array([args[i] for i in range(len(args)//2+1,len(args))])
#     return np.math.factorial(n)/(np.math.factorial(x)*np.math.factorial(n-x)) * (p**x) * ((1-p)**(n-x))
# Multi = np.vectorize(Multi)
#
#
if __name__ == "__main__":
    Dis = BetaDistribution(7,5)
    print(Dis.distribution(0.9))
    X = np.array([[1,1],
                  [1,2],
                  [1,3]])
    Y = np.array([2,
                  6,
                  4])
    from numpy.linalg import inv
    theta = np.matmul(inv(np.matmul(np.transpose(X),X)), np.matmul(np.transpose(X),Y))
    print(theta)
#     import matplotlib.pyplot as plt
#     # Dis = MultinomialDistribution(100,.2)
#     X = np.array(list(range(1,100)))
#     Y = Multi(X)
#     plt.plot(X,Y,'o')
#     plt.show()
