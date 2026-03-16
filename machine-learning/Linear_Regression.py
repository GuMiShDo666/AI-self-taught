from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

def linearRegression(alpha=0.01,num_iters=400):
    print(u"加载数据...\n")
    
    data = loadtxtAndcsv_data("data.txt",",",np.float64) 
    X = data[:,0:-1]                        
    y = data[:,-1]          
    m = len(y)            
    col = data.shape[1]      
    
    X,mu,sigma = featureNormaliza(X)    # 归一化
    plot_X1_X2(X)         
    
    X = np.hstack((np.ones((m,1)),X))    
    
    print(u"\n执行梯度下降算法....\n")
    
    theta = np.zeros((col,1))
    y = y.reshape(-1,1)   
    theta,J_history = gradientDescent(X, y, theta, alpha, num_iters)
    
    plotJ(J_history, num_iters)
    
    return mu,sigma,theta   #返回均值mu,标准差sigma,和学习的结果theta
    
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

def loadnpy_data(fileName):
    return np.load(fileName)

def featureNormaliza(X):
    X_norm = np.array(X)     
    mu = np.zeros((1,X.shape[1]))   
    sigma = np.zeros((1,X.shape[1]))
    
    mu = np.mean(X_norm,0)
    sigma = np.std(X_norm,0)
    for i in range(X.shape[1]):
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i] 
    
    return X_norm,mu,sigma

def plot_X1_X2(X):
    plt.scatter(X[:,0],X[:,1])
    plt.show()

# 梯度下降算法
def gradientDescent(X,y,theta,alpha,num_iters):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)

    m = len(y)      
    n = len(theta)
    
    temp = np.matrix(np.zeros((n,num_iters)))   # 暂存每次迭代计算的theta，转化为矩阵形式
    
    J_history = np.zeros((num_iters,1)) #记录每次迭代计算的代价值
    
    for i in range(num_iters):   
        h = X * theta
        temp[:,i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y)))   #梯度的计算
        theta = temp[:,i]
        J_history[i] = computerCost(X,y,theta)      #调用计算代价函数
        print('.', end=' ')      
    return theta,J_history  

# 计算代价函数
def computerCost(X,y,theta):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)

    m = len(y)
    J = 0
    
    J = (np.transpose(X*theta-y))*(X*theta-y)/(2*m)
    return J

def plotJ(J_history,num_iters):
    x = np.arange(1,num_iters+1)
    plt.plot(x,J_history)
    plt.xlabel(u"迭代次数") 
    plt.ylabel(u"代价值")
    plt.title(u"代价随迭代次数的变化")
    plt.show()

def testLinearRegression():
    mu,sigma,theta = linearRegression(0.01,400)
    #print u"\n计算的theta值为：\n",theta
    #print u"\n预测结果为：%f"%predict(mu, sigma, theta)
    
def predict(mu,sigma,theta):
    result = 0
    predict = np.array([1650,3])
    norm_predict = (predict-mu)/sigma
    final_predict = np.hstack((np.ones((1,1)),norm_predict))
    
    result = np.dot(final_predict,theta)
    return result
    
    
if __name__ == "__main__":
    testLinearRegression()
