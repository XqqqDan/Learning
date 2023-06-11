import numpy as np
import math
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import linear_sum_assignment

def myBestMap(trueLabels, estLabels):

    Inf = math.inf

    trueLabelVals = np.unique(trueLabels)
    kTrue = len(trueLabelVals)
    estLabelVals = np.unique(estLabels)
    kEst = len(estLabelVals)

    cost_matrix = np.zeros([kEst, kTrue])
    for ii in range(kEst):
        inds = np.where(estLabels == estLabelVals[ii])
        for jj in range(kTrue):
            cost_matrix[ii,jj] = np.size(np.where(trueLabels[inds] == trueLabelVals[jj]))
    
    rInd, cInd = linear_sum_assignment(-cost_matrix)

    outLabels = Inf * np.ones(np.size(estLabels)).reshape(np.size(trueLabels), 1)

    for ii in range(rInd.size):
        outLabels[estLabels == estLabelVals[rInd[ii]]] = trueLabelVals[cInd[ii]]

    outLabelVals = np.unique(outLabels)
    if np.size(outLabelVals) < max(outLabels):
        lVal = 1
        for ii in range(np.size(outLabelVals)):
            outLabels[outLabels == outLabelVals[ii]] = lVal
            lVal += 1       
    return outLabels
    
def missRate(trueLabels, estLabels):
    estLabels = myBestMap(trueLabels, estLabels)
    err = np.sum(trueLabels != estLabels) / np.size(trueLabels)

    return err, estLabels
