import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MatrixMethods
from Camera import Camera
from SingleImage import SingleImage
import cv2

def is_orthonormal(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Calculate the dot product of each pair of columns
    dot_product_matrix = np.dot(matrix.T, matrix)

    # Check if the dot product matrix is an identity matrix
    identity_matrix = np.eye(matrix.shape[1])
    if not np.allclose(dot_product_matrix, identity_matrix):
        return False

    for i in range(matrix.shape[1] - 1):
        for j in range(i + 1, matrix.shape[1]):
            dot_product = np.dot(matrix[:, i], matrix[:, j])
            if not np.isclose(dot_product, 0):
                return False
    return True

def CalculateCrossPoint(Points1, Points2):
    m1 = (Points1[0, 1] - Points1[1, 1]) / (Points1[0, 0] - Points1[1, 0])
    b1 = Points1[1, 1] - m1 * Points1[1, 0]

    m2 = (Points2[0, 1] - Points2[1, 1]) / (Points2[0, 0] - Points2[1, 0])
    b2 = Points2[1, 1] - m2 * Points2[1, 0]

    x = (b1 - b2) / (m2 - m1)
    y = m1 * x + b1


    return np.array([[x,y,1]]).T

############## Create Syntetic Image ###############################
PointsListX = []
PointsListY = []
for i in range(10):
    PointsListX.append([i, 2.5, 0])
    PointsListX.append([i, 7.5, 0])

    PointsListY.append([2.5, i, 0])
    PointsListY.append([7.5, i, 0])

PointsListX = np.array(PointsListX)
PointsListY = np.array(PointsListY)

C1 = Camera(1000, [500,500], [], [], [],2000)
Image1 = SingleImage(C1)
Image1.exteriorOrientationParameters={"X0":4.5,"Y0":-5,"Z0":1.5,"omega":85*np.pi/180,"phi":15*np.pi/180,"kappa":0.5*np.pi/180}

ImgPointsY = Image1.GroundToImage(PointsListY)
ImgPointsX = Image1.GroundToImage(PointsListX)
ImgPointsY = np.concatenate((ImgPointsY, np.ones((len(ImgPointsY),1))),axis=1)
ImgPointsX = np.concatenate((ImgPointsX, np.ones((len(ImgPointsX),1))),axis=1)
################################################################################################################################################

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(PointsListX[:,0], PointsListX[:,1], c="blue")
ax1.scatter(PointsListY[:,0], PointsListY[:,1], c="g")

ax2.scatter(ImgPointsX[:,0], ImgPointsX[:,1], c="blue")
ax2.scatter(ImgPointsY[:,0], ImgPointsY[:,1], c="g")
ax2.plot([-1000,-1000,1000,1000,-1000],[1000,-1000,-1000,1000,1000], c="black")
#####################################################################################################################################################################

V1 = Image1.findVanishingPoint(ImgPointsX[::2,:],ImgPointsX[1::2,:])
V2 = Image1.findVanishingPoint(ImgPointsY[::2,:],ImgPointsY[1::2,:])

ax2.scatter(V1[0,:], V1[1,:])
ax2.scatter(V2[0,:], V2[1,:])

###### calcolate the cross point and compare#################################################################
Vcalculate = CalculateCrossPoint(ImgPointsY[[0,-2],:],ImgPointsY[[1,-1],:])

Line1 = np.concatenate((ImgPointsY[[-2],:],Vcalculate.T))
Line2 = np.concatenate((ImgPointsY[[-1],:],Vcalculate.T))

ax2.plot(Line1[:,0],Line1[:,1])
ax2.plot(Line2[:,0],Line2[:,1])

MatrixMethods.PrintMatrix(Vcalculate - V2)

Vcalculate = CalculateCrossPoint(ImgPointsX[[0,-2],:],ImgPointsX[[1,-1],:])

Line1 = np.concatenate((ImgPointsX[[-2],:],Vcalculate.T))
Line2 = np.concatenate((ImgPointsX[[-1],:],Vcalculate.T))

ax2.plot(Line1[:,0],Line1[:,1])
ax2.plot(Line2[:,0],Line2[:,1])

MatrixMethods.PrintMatrix(Vcalculate - V1)
plt.show()



Dgimut = pd.read_csv("Dgimut.csv").to_numpy()
################# Calculate for real Image ##################################################

RealImage = SingleImage(C1)

V1_1 = RealImage.findVanishingPoint(np.concatenate((Dgimut[:,:2],np.ones((2,1))),axis = 1),np.concatenate((Dgimut[:,2:4],np.ones((2,1))),axis = 1))
V1_2 = RealImage.findVanishingPoint(np.concatenate((Dgimut[:,2:4],np.ones((2,1))),axis = 1),np.concatenate((Dgimut[:,4:6],np.ones((2,1))),axis = 1))
V1_3 = RealImage.findVanishingPoint(np.concatenate((Dgimut[:,:2],np.ones((2,1))),axis = 1),np.concatenate((Dgimut[:,4:6],np.ones((2,1))),axis = 1))

V1diff = np.concatenate((V1_1,V1_2,V1_3),axis = 1)
V1avg = np.average(V1diff,1)
MatrixMethods.PrintMatrix(V1diff)
MatrixMethods.PrintMatrix(np.max(V1diff,1)-np.min(V1diff,1))

V2_1 = RealImage.findVanishingPoint(np.concatenate((Dgimut[:,6:8],np.ones((2,1))),axis = 1),np.concatenate((Dgimut[:,8:10],np.ones((2,1))),axis = 1))
V2_2 = RealImage.findVanishingPoint(np.concatenate((Dgimut[:,8:10],np.ones((2,1))),axis = 1),np.concatenate((Dgimut[:,10:],np.ones((2,1))),axis = 1))
V2_3 = RealImage.findVanishingPoint(np.concatenate((Dgimut[:,6:8],np.ones((2,1))),axis = 1),np.concatenate((Dgimut[:,10:],np.ones((2,1))),axis = 1))

V2diff = np.concatenate((V2_1,V2_2,V2_3),axis = 1)
V2avg = np.average(V2diff,1)
MatrixMethods.PrintMatrix(V2diff)
MatrixMethods.PrintMatrix(np.max(V2diff,1)-np.min(V2diff,1))



 #####################Image show ##############################################

Borowitz = cv2.imread("Borowitz.jpg")
Borowitz = cv2.cvtColor(Borowitz, cv2.COLOR_BGR2RGB)

plt.imshow(Borowitz)
X = Dgimut[:,0:6:2].reshape(-1,)
Y = Dgimut[:,1:6:2].reshape(-1,)
plt.scatter(X,Y)
plt.scatter(V1avg[0],V1avg[1], c="black",s=50)


X = Dgimut[:,6::2].reshape(-1,)
Y = Dgimut[:,7::2].reshape(-1,)
plt.scatter(X,Y)
plt.scatter(V2_3[0],V2_3[1], c="black", s=50)

###X axis ###################################################################

VcalculateimgX = CalculateCrossPoint(Dgimut[[0,1],:2],Dgimut[[0,1],2:4])
Line1 = np.concatenate((np.concatenate((Dgimut[1,:2],np.ones(1))).reshape((1,3)),VcalculateimgX.T))
Line2 = np.concatenate((np.concatenate((Dgimut[1,2:4],np.ones(1))).reshape((1,3)),VcalculateimgX.T))

plt.plot(Line1[:,0],Line1[:,1])
plt.plot(Line2[:,0],Line2[:,1])

###Y axis ###################################################################
VcalculateimgY = CalculateCrossPoint(Dgimut[[0,1],6:8],Dgimut[[0,1],10:])

Line1 = np.concatenate((np.concatenate((Dgimut[1,6:8],np.ones(1))).reshape((1,3)),VcalculateimgY.T))
Line2 = np.concatenate((np.concatenate((Dgimut[1,10:],np.ones(1))).reshape((1,3)),VcalculateimgY.T))

plt.plot(Line1[:,0],Line1[:,1])
plt.plot(Line2[:,0],Line2[:,1])
plt.show()

####### Part 2 ###########################################################################
L1_1 = pd.read_csv("L1_1.csv", header=None).to_numpy()
L1_2 = pd.read_csv("L1_2.csv", header=None).to_numpy()
L2_1 = pd.read_csv("L2_1.csv", header=None).to_numpy()
L2_2 = pd.read_csv("L2_2.csv", header=None).to_numpy()

Image = cv2.imread("IMG_2977.jpeg")

NewImage = Image1.ProjectiveCanclation(L1_1,L1_2,L2_1,L2_2, Image)

plt.imshow(NewImage)
plt.show()

x=0
