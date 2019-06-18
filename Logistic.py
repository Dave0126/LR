import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

path = "iris.csv"

def cutinfo(path):
    iris_info = pd.read_csv(path)
    iris_info.loc[iris_info.Species == 'setosa', 'Species'] = 0  # setosa归类为1
    iris_info.loc[iris_info.Species == 'versicolor', 'Species'] = 1  # versicolor归类为0

    setosa = iris_info.iloc[:50, :] #构造数据集,0~50行
    versicolor = iris_info.iloc[50:100, :]  #51~100行
    virginica = iris_info.iloc[100:, :]     #101~150行   #不用这种花
    '''
    print(type(setosa))
    print(type(versicolor))
    print(type(virginica))
    '''
    iris_traning = pd.concat([setosa.sample(frac=0.3), versicolor.sample(frac=0.3)], axis=0)  # 随机取两种花的做训练集，列对齐
    iris_info = iris_info.iloc[0:100, :]    #总的数据集
    iris_test = iris_info.append(iris_traning)  #合并，训练集在下

    iris_test = iris_test.drop_duplicates(subset=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'], keep=False)#求差集
    # print(type(iris_test))
    iris_info = pd.concat([iris_info.iloc[:, 2:3],iris_info.iloc[:, 4:]], axis=1)
    iris_traning = pd.concat([iris_traning.iloc[:, 2:3],iris_traning.iloc[:, 4:]], axis=1)
    iris_test = pd.concat([iris_test.iloc[:, 2:3],iris_test.iloc[:, 4:]], axis=1)

    iris_info.insert(0, 'Ones', 1)
    iris_traning.insert(0, 'Ones', 1)
    iris_test.insert(0, 'Ones', 1)
    #print(type(iris_traning))


    iris_traning = pd.DataFrame(iris_traning, dtype=np.float)#将object转换为float64
    traning_data = iris_traning.values
    iris_test = pd.DataFrame(iris_test, dtype=np.float)#将object转换为float64
    test_data = iris_test.values
    # print(type(traning_data))

    return traning_data,test_data,iris_traning,iris_test
#打乱
def upset(data):
    np.random.shuffle(data)
    ind = data.shape[1]
    X = data[:,0:ind-1]
    y = data[:,ind-1:]
    return X,y

def sigmoid(w): #sigmoidFunction
    return 1/(1 + np.exp(-w))

def prediction(X,theta): #预测函数
    return sigmoid(np.dot(X,theta))

def predict(X,theta):
    '''
    for x in model(X, theta):
        print(x)
    '''
    return [1 if x >= 0.5 else 0 for x in prediction(X, theta)]
'''
print(theta)
print(X[:5])
print(y[:5])
'''
#损失函数
def costFunction(X,y,theta):
    return np.sum(np.multiply(-y,np.log(prediction(X,theta)))-np.multiply(1-y,np.log(1-prediction(X,theta))))/(len(X))

#梯度下降求解
def gradientDescent(data,batchSize,thresh,alpha):
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #batch
    ind = data.shape[1]
    X = data[:, 0:ind - 1]
    y = data[:, ind - 1:]
    X,y = upset(data)
    theta = np.mat(np.random.randn(3,1))#theta随机初始化
    print('初始的theta向量：\n{0}'.format(theta))
    costs = [] #损失值
    while True:
        P = prediction(X[k:k+batchSize],theta)#(10,1)
        dt = X[k:k+batchSize].T*(P-y[k:k+batchSize])#求梯度
        k += batchSize #取batch数量个数据
        if k >= n: #大于数据个数
            k = 0
            X,y = upset(data) #重新打乱
        #print(theta)
        theta -=  alpha*dt #更新theta
        costs.append(costFunction(X,y,theta)) #计算新损失

        i += 1
        if i > thresh:
            print('迭代的costs值为：\n{0}'.format(costs))
            break

    return theta,i-1,costs,time.time() - init_time
def Running_cost(data,batchSize,thresh,alpha):
    theta,iter,costs,dur = gradientDescent(data,batchSize,thresh,alpha)
    print('\n结果的theta向量化表示:\n{0}\n\n最后的cost值为:{1}'.format(theta,costs[-1]))#取最新的theta和cost

    fig,ax = plt.subplots()
    ax.plot(np.arange(len(costs)),costs,'b')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    plt.show()
    return theta


n = 25  #batchsize
traning_data,test_data,iris_traning,iris_test = cutinfo(path)
theta = Running_cost(traning_data,n,thresh=5000,alpha=0.001)

# 绘制散点图
positive = iris_test[iris_test['Species'] == 1]
negative = iris_test[iris_test['Species'] == 0]
fig, ax = plt.subplots()
ax.scatter(positive['Sepal.Width'], positive['Petal.Width'], c='y', marker='*', label='setosa')
ax.scatter(negative['Sepal.Width'], negative['Petal.Width'], c='g', marker='x', label='versicolor')
ax.legend()
ax.set_xlabel('Sepal.Width')
ax.set_ylabel('Petal.Width')
# 绘制决策边界
a = theta[0,0]
b = theta[1,0]
c = theta[2,0]
x1 = np.arange(1,8,0.1)
x2 = -a/c-b/c*x1
plt.plot(x1,x2)
plt.show()

scaled_X = test_data[:,:3]
y = test_data[:,3]
predictions = predict(scaled_X,theta)
correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(correct) / len(correct))
print('正确率 = {:.5f}'.format(accuracy))
for (a, b) in zip(predictions, y):
    print((a,b))