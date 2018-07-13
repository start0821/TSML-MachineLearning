# Normal Distribution
import math
import numpy as np


class NormalDistribution:
    def __init__(self,average=0,variance=1):
        self.average = average
        self.variance = variance
        self.distribution = np.vectorize(self.distribution)

    def distribution(self,x=0):
        # if type(x) is np.ndarray:
        #     average_array = np.array(x.size)
        #     average_array.fill(self.average)
        #     return (1./(self.variance*math.sqrt(math.pi*2.)))*math.exp(-(x-average_array)**2/(2.*self.variance**2))
        return (1./(self.variance*math.sqrt(math.pi*2.)))*math.exp(-(x-self.average)**2/(2.*self.variance**2))

def N(x,average=0,variance=1):
    # if type(x) is np.ndarray:
    #     # average_array = np.array(x.size)
    #     # average_array.fill(average)
    #     return (1./(variance*math.sqrt(math.pi*2.)))*math.exp(-(x-average)**2/(2.*variance**2))
    return (1./(variance*math.sqrt(math.pi*2.)))*math.exp(-(x-average)**2/(2.*variance**2))
N = np.vectorize(N)


class BetaDistribution:
    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        self.B = math.gamma(self.alpha)*math.gamma(self.beta)/math.gamma(self.beta+self.alpha)
        self.distribution = np.vectorize(self.distribution)

    def distribution(self,x):
        return (x**self.alpha*(1-x)**self.beta)/self.B

def Beta(x,alpha,beta):
    B = math.gamma(alpha)*math.gamma(beta)/math.gamma(beta+alpha)
    return (x**alpha*(1-x)**beta)/B
Beta = np.vectorize(Beta)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Dis = BetaDistribution(.5,.5)
    X = np.array(list(range(0,1000)))/1000
    Y = Dis.distribution(X)
    plt.plot(X,Y)
    plt.show()
    print(Dis.distribution(.2))
