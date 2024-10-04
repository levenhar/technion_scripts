import pandas as pd
from Reader import Reader as R
import Camera as Cam
from SingleImage import SingleImage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import MatrixMethods as MM
from ImagePair import ImagePair
import PhotoViewer as PV
from ImageTriple import ImageTriple

def digitleImageToCamera(image,Ipoints):
    '''

    :param image: image object which the points had been sampled
    :type image:
    :param points: points array to convert
    :type points: array (nx2)
    :return:points in the camera system
    :rtype:array (nx2)
    '''
    S=np.array([[1,0],[0,-1]])
    xyP=image.camera.principalPoint
    xyP=np.reshape(xyP,(2,1))
    Ipoints=Ipoints.T
    Cpoints=-S@xyP+S@(Ipoints)
    image.isSolved=True
    return Cpoints.T

#Camera Properties
f=(4627.85368+4619.59892)/2
xp=2741.63382
yp=1711.03851

#Create Camera Object
Cam1=Cam.Camera(f,np.array([xp,yp]),np.array([]),np.array([]),None)

#Create ImageObjects
Image1=SingleImage(Cam1) #SingleImage object for 09 Image
Image2=SingleImage(Cam1)  #SingleImage object for 10 Image
Image2.exteriorOrientationParameters={"X0":0,"Y0":0,"Z0":0}
Image3=SingleImage(Cam1)  #SingleImage object for 11 Image

#Reading the Data
points1=pd.read_csv("3509.csv",header=None).to_numpy()
points2=pd.read_csv("3510.csv",header=None).to_numpy()
points3=pd.read_csv("3511.csv",header=None).to_numpy()

#convert the points to ideal Camera
CamPoints1=digitleImageToCamera(Image1,points1)
CamPoints2=digitleImageToCamera(Image2,points2)
CamPoints3=digitleImageToCamera(Image3,points3)

#print the points
'''MM.PrintMatrix(np.concatenate((points1,CamPoints1),axis=1))
MM.PrintMatrix(np.concatenate((points2,CamPoints2),axis=1))
MM.PrintMatrix(np.concatenate((points3,CamPoints3),axis=1))'''

#Create model object
model1=ImagePair(Image1,Image2)
model2=ImagePair(Image2,Image3)


initialParam=np.array([[0,0,0,0,0]],dtype="float64")
initialParam=np.reshape(initialParam,(5,1))

#remove the points that does not exists in both images
CamPoints1Model1=np.copy(CamPoints1)
CamPoints1Model1=np.delete(CamPoints1Model1, [0,5,13,25], 0)
CamPoints2Model1=np.copy(CamPoints2)
CamPoints2Model1=np.delete(CamPoints2Model1, [0,5,13,25], 0)

CamPoints2Model2=np.copy(CamPoints2)
CamPoints2Model2=np.delete(CamPoints2Model2, [0,12,13,25,30,31,36,37,39,43,44,48], 0)
CamPoints3Model2=np.copy(CamPoints3)
CamPoints3Model2=np.delete(CamPoints3Model2, [0,12,13,25,30,31,36,37,39,43,44,48], 0)


#Calculate the Relative Orientation
Image2Parm, v1, sigma1, Sx1,Q1 = model1.ComputeDependentRelativeOrientation(CamPoints1Model1,CamPoints1Model1,initialParam)

initialParam=np.array([[0,0,0,0,0]],dtype="float64")
initialParam=np.reshape(initialParam,(5,1))

Image3Parm, v2, sigma2, Sx2, Q2= model2.ComputeDependentRelativeOrientation(CamPoints2Model2,CamPoints3Model2,initialParam)

#print the result
'''print(model1.PerspectiveCenter_Image2)
MM.PrintMatrix(model1.RotationMatrix_Image2)
print(np.diag(Sx1))

print(model2.PerspectiveCenter_Image2)
MM.PrintMatrix(model2.RotationMatrix_Image2)
print(np.diag(Sx2))'''


#### Section C - Model 2

CamPoints3Model2=pd.read_csv("3510Model.csv",header=None).to_numpy()
CamPoints2Model2=pd.read_csv("3511Model.csv",header=None).to_numpy()
CamPoints2M=digitleImageToCamera(Image2,CamPoints3Model2)
CamPoints3M=digitleImageToCamera(Image3,CamPoints2Model2)

#Calculate the points by vector method
ModelPoints=model2.vectorIntersction(CamPoints2M,CamPoints3M)
ModelArray=np.array([])

#Convert the dict to Array
for i in range(len(ModelPoints)//2):
    ModelArray=np.concatenate((ModelArray,np.array(ModelPoints[f"P{i}"])),axis=0)
ModelArray=np.reshape(ModelArray,(len(ModelPoints)//2,3))


#3D - Draw
'''model2.drawImagePair(ModelArray)
plt.show()'''

#print the results
MM.PrintMatrix(ModelArray)

#----------------------Lab8------------------------Lab8-----------------------

print("Lab 8 - Results")
#Orientation withour Scale
b31=model1.PerspectiveCenter_Image2+model1.RotationMatrix_Image2@model2.PerspectiveCenter_Image2
R31=model1.RotationMatrix_Image2@model2.RotationMatrix_Image2

omega=np.arctan2(-R31[1,2],R31[2,2])
kappa=np.arctan2(-R31[0,1],R31[0,0])
phi=np.arcsin(R31[0,2])

print(b31)
print([omega,phi,kappa])


CamPoints1Model1=pd.read_csv("3509Model.csv",header=None).to_numpy()
CamPoints2Model1=pd.read_csv("3511Model.csv",header=None).to_numpy()
CamPoints1Model1=np.delete(CamPoints1Model1, [6], 0)
CamPoints2Model1=np.delete(CamPoints2Model1, [6], 0)


CamPoints1M=digitleImageToCamera(Image1,CamPoints1Model1)
CamPoints2M=digitleImageToCamera(Image2,CamPoints2Model1)

#Calculate the points by vector method
ModelPoints=model2.vectorIntersction(CamPoints1M,CamPoints2M)
ModelArray2=np.array([])

#Convert the dict to Array
for i in range(len(ModelPoints)//2):
    ModelArray2=np.concatenate((ModelArray2,np.array(ModelPoints[f"P{i}"])),axis=0)
ModelArray2=np.reshape(ModelArray2,(len(ModelPoints)//2,3))

#print the results
MM.PrintMatrix(ModelArray2)

#################################################################################################
FullModel=ImageTriple(model1,model2)

'''FullModel.drawModles(FullModel.imagePair1,FullModel.imagePair2,ModelArray,ModelArray2)
plt.show()'''

#calculate the scale factor between the models
scales=[]
for i in range(11):
    P1=np.array([[CamPoints1M[i,0]],[CamPoints1M[i,1]]])
    P2=np.array([[CamPoints2M[i,0]],[CamPoints2M[i,1]]])
    P3=np.array([[CamPoints3M[i,0]],[CamPoints3M[i,1]]])
    S=FullModel.ComputeScaleBetweenModels(P1,P2,P3)
    scales.append(S)

Avg_Scale=np.average(scales)
std_Scale=np.std(scales)

print(Avg_Scale)
print(std_Scale)


O2=FullModel.imagePair1.PerspectiveCenter_Image2
b32=FullModel.imagePair2.PerspectiveCenter_Image2
R2=FullModel.imagePair1.RotationMatrix_Image2

#calculate the prespective center of Image 3
O3=O2+Avg_Scale*R2@b32
print(O3)

Model3P=FullModel.RayIntersection(CamPoints1M,CamPoints2M,CamPoints3M)
MM.PrintMatrix(Model3P)

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plot the Model points
ax.scatter(Model3P[:,0],Model3P[:,1],Model3P[:,2],'red')

#Plot the Model edges
ax.plot(Model3P[:3,0],Model3P[:3,1],Model3P[:3,2],"blue")
ax.plot(Model3P[4:6, 0], Model3P[4:6, 1], Model3P[4:6, 2],"blue")
ax.plot([Model3P[7, 0],Model3P[4, 0]], [Model3P[8, 1],Model3P[4, 1]], [Model3P[7, 2],Model3P[4, 2]],"blue")
ax.plot([Model3P[8, 0], Model3P[9, 0],Model3P[10, 0]], [Model3P[8, 1], Model3P[9, 1],Model3P[10, 1]],[Model3P[8, 2], Model3P[9, 2],Model3P[10, 2]],"blue")

#plt.show()
'''

#--------------------Lab9-------------------------Lab9----------------------------------------------------

#calculate Normal Vector
#A
LevelPoints=Model3P[-6:-3,:]

V1=(LevelPoints[1,:]-LevelPoints[0,:]).reshape((3))
V2=(LevelPoints[2,:]-LevelPoints[0,:]).reshape((3))

NV=np.cross(V1,V2)
NV=NV/np.linalg.norm(NV)
print(NV)

V1=V1/np.linalg.norm(V1)
R=FullModel.imagePair1.RotationLevelModel(("z",NV),("x",V1))
print("the Rotation Matrix from 1,7,17")
MM.PrintMatrix(R)

#B
LevelPoints2=Model3P[-3:,:]

V1=(LevelPoints2[1,:]-LevelPoints2[0,:]).reshape((3))
V2=(LevelPoints2[2,:]-LevelPoints2[0,:]).reshape((3))

NV=np.cross(V1,V2)
NV=NV/np.linalg.norm(NV)

V1=V1/np.linalg.norm(V1)
R=FullModel.imagePair1.RotationLevelModel(("z",NV),("x",V1))
print("the Rotation Matrix from 5,3,13")
MM.PrintMatrix(R)

#-----Scale Factor
l78=0.117
l85=0.47
l13=0.47

p7=Model3P[6,:].reshape((3,1))
p8=Model3P[7,:].reshape((3,1))
p5=Model3P[4,:].reshape((3,1))
p3=Model3P[2,:].reshape((3,1))
p1=Model3P[0,:].reshape((3,1))

l78M=np.linalg.norm(p7-p8)
l85M=np.linalg.norm(p5-p8)
l13M=np.linalg.norm(p1-p3)

S1=l78/l78M
S2=l85/l85M
S3=l13/l13M

S=np.average([S1,S2,S3])
stdS=np.std([S1,S2,S3])

print(S)
print(stdS)

RealPoints=FullModel.imagePair1.ModelTransformation(Model3P,S/10)

p9=RealPoints[8,:].reshape((3,1))
p10=RealPoints[9,:].reshape((3,1))
p11=RealPoints[10,:].reshape((3,1))
p6=RealPoints[5,:].reshape((3,1))
p5=RealPoints[4,:].reshape((3,1))
p1=RealPoints[0,:].reshape((3,1))
p2=RealPoints[2,:].reshape((3,1))

l910R=np.linalg.norm(p9-p10)
l1011R=np.linalg.norm(p10-p11)
l65R=np.linalg.norm(p5-p6)
l13R=np.linalg.norm(p1-p3)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plot the Model points
ax.scatter(Model3P[:,0],Model3P[:,1],Model3P[:,2],'red')

#Plot the Model edges
ax.plot(Model3P[:3,0],Model3P[:3,1],Model3P[:3,2],"blue")
ax.plot(Model3P[4:6, 0], Model3P[4:6, 1], Model3P[4:6, 2],"blue")
ax.plot([Model3P[7, 0],Model3P[4, 0]], [Model3P[8, 1],Model3P[4, 1]], [Model3P[7, 2],Model3P[4, 2]],"blue")
ax.plot([Model3P[8, 0], Model3P[9, 0],Model3P[10, 0]], [Model3P[8, 1], Model3P[9, 1],Model3P[10, 1]],[Model3P[8, 2], Model3P[9, 2],Model3P[10, 2]],"blue")

#plot the model in the right scale
ax.scatter(RealPoints[:,0],RealPoints[:,1],RealPoints[:,2],'black')
ax.plot(RealPoints[:3,0],RealPoints[:3,1],RealPoints[:3,2],"black")
ax.plot(RealPoints[4:6, 0], RealPoints[4:6, 1], RealPoints[4:6, 2],"black")
ax.plot([RealPoints[7, 0],RealPoints[4, 0]], [RealPoints[8, 1],RealPoints[4, 1]], [RealPoints[7, 2],RealPoints[4, 2]],"black")
ax.plot([RealPoints[8, 0], RealPoints[9, 0],RealPoints[10, 0]], [RealPoints[8, 1], RealPoints[9, 1],RealPoints[10, 1]],[RealPoints[8, 2], RealPoints[9, 2],RealPoints[10, 2]],"black")
plt.show()
