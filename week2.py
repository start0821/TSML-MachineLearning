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
Bi = np.vectorize(Beta)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Dis = BinomialDistribution(100,.2)
    X = np.array(list(range(1,100)))
    Y = Dis.distribution(X)
    plt.plot(X,Y,'o')
    plt.show()
