import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse.linalg import svds
from sklearn import decomposition
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def MyPCA(data):
    """
    Syntax:       dataR = MyPCA(data)

    Inputs:
    -------
        data is an (M x N) matrix, waiting for PCA process.

    Outputs:
    -------
        dataR is the datasets after PCA process.

    Description:
    -------
        The function can reduce the demention of dataset`s characteristics, reduce to 2 demention, which is remain the biggest eigenvalue and the successive eigenvalue. 
    """

    mean = np.mean(data, axis=1)

    for i in range(len(data[1])):
        data[:, i] =data[:, i] - mean

    U, sigma, VT = svds(data)
    print(sigma)

    u = np.vstack((U[:, 0].T, U[:, 1].T)).T
    v = np.vstack((VT[0], VT[1]))

    dataPca = u @ np.diag([sigma[0], sigma[1]]) @ v

    for i in range(len(dataPca[0])):
        dataPca[:, i] = dataPca[:, i] + mean

    return dataPca

def standardize(data1, data2):
    """
    Syntax:       dataR = standardize(data)

    Inputs:
    -------
        data is an (M x N) matrix, waiting for standardizition.

    Outputs:
    -------
        dataR is an (N x M) matrix is the datasets after standardizition.

    Description:
    -------
        The function can standardize the data for the ridge regression. 
    """
    transfer = StandardScaler()

    dataR1 = transfer.fit_transform(data1)
    dataR1 = dataR1.T

    dataR2 = transfer.fit_transform(data2)
    dataR2 = dataR2.T 
    
    return dataR1, dataR2


# load data
data = loadmat("mnistSubset.mat")
X = data['X']
trueLabels = data['trueLabels'][:,0]

# only consider digits 1 and 2 (labels 2 and 3)
K = 2
Xtrain = np.array([]).reshape(784, 0)
ytrain = np.array([]).reshape(0, 1)
Xtest = np.array([]).reshape(784, 0)
ytest = np.array([]).reshape(0, 1)
for kk in range(1,K+1):
    Xk = X[:, np.argwhere(trueLabels==kk+1)[:,0]]
    Xtrain = np.append(Xtrain, Xk[:, :100], axis=1)
    Xtest = np.append(Xtest, Xk[:, 100:], axis=1)
    ytrain = np.append(ytrain, kk*np.ones([100, 1]), axis=0)
    ytest = np.append(ytest, kk*np.ones([100, 1]), axis=0)

# make labels +1/-1
ytrain = (2*(ytrain - 1) - 1)[:,0]
ytest = (2*(ytest - 1) - 1)[:,0]

# ridge regression parameter
lam = 1e-3

## YOUR CODE

# train LS classifier on Xtrain/ytrain

# standardize the train data and test data
XtrainH, XtestH = standardize(Xtrain, Xtest)

# the ridge regression
estimator = Ridge(alpha=1)
# train LS classifier
estimator.fit(XtrainH, ytrain)
wH = estimator.coef_

# report both training and test error
ypredict = np.sign(estimator.predict(XtrainH))
error = mean_squared_error(ytrain, ypredict)
print("Train datasets error:", error)

ypredict = np.sign(estimator.predict(XtestH))
error = mean_squared_error(ytest, ypredict)
print("Test datasets error:", error)
# print("Prediction:\n", ypredict)
# print("Testvalue:\n", ytest)


# reduce dimension of data using PCA and plot

# reduce dimension of datasets using PCA
XtrainL = MyPCA(Xtrain)
XtestL = MyPCA(Xtest)

# standardize the train data and test data
XtrainL, XtestL = standardize(Xtrain, Xtest)

# classify dimension-reduced data using least squares

# the ridge regression
estimator = Ridge(alpha=1)
# train LS classifier
estimator.fit(XtrainL, ytrain)
wL = estimator.coef_

ypredict = np.sign(estimator.predict(XtrainL))
error = mean_squared_error(ytrain, ypredict)
print("Train datasets error:", error)

ypredict = np.sign(estimator.predict(XtestL))
error = mean_squared_error(ytest, ypredict)
print("Test datasets error:", error)
# print("Prediction:\n", ypredict)
# print("Testvalue:\n", ytest)

## plot LS separator with data
minX = np.min(XtrainL[0,:])
maxX = np.max(XtrainL[0,:])
xx = np.linspace(minX, maxX,100)
yy = -wL[0]*xx/wL[1]
plt.plot(xx,yy, label='LS separator')
plt.legend()
plt.show()
