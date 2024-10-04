from DLT import DLT
import MatrixMethods as MM
from Camera import Camera
from SingleImage import SingleImage
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import PhotoViewer


################### Create Syntetic Data #######################################################################################
camera1=Camera(1536, [994,1012], [], [], [],2000)

Image1=SingleImage(camera1)
Image1.exteriorOrientationParameters={"X0":5,"Y0":4,"Z0":15,"omega":4*np.pi/180,"phi":5*np.pi/180,"kappa":3*np.pi/180}

n=int(np.floor(10*np.random.random()+10))

IS=Image1.IamgeSignature()
PolyS = Polygon(IS)
min_x, min_y, max_x, max_y = PolyS.bounds
ControlPoints = []
while len(ControlPoints) < n:
    x = np.random.uniform(min_x, max_x)
    y = np.random.uniform(min_y, max_y)
    P1 = Point(x, y)
    if PolyS.contains(P1):
        ControlPoints.append([x, y, 1*np.random.random()-1])
ControlPoints = np.array(ControlPoints)

cameraPoints = Image1.GroundToImage(ControlPoints)
ImagePoints = camera1.Cam2Img(cameraPoints)
#ImagePoints=np.array(list(map(lambda P:P+Image1.camera.principalPoint,cameraPoints)))
################################################################################################################################

######## Plot the Data ########################################################################################################
fig=plt.figure()
ax=fig.add_subplot(121, projection='3d')
ax.scatter(ControlPoints[:,0],ControlPoints[:,1],ControlPoints[:,2])
IS = np.concatenate((IS,IS[0,:].reshape((1,3))))
IS[:,2] -=1
ax.plot(IS[:,0],IS[:,1],IS[:,2],c="g")

X0=np.array([[Image1.exteriorOrientationParameters["X0"],Image1.exteriorOrientationParameters["Y0"],Image1.exteriorOrientationParameters["Z0"]]]).T
PhotoViewer.drawImageFrame(Image1.camera.SensorSize,Image1.camera.SensorSize,Image1.rotationMatrix,X0,Image1.camera.focalLength,0.005,ax)

ax1=fig.add_subplot(122)
ax1.scatter(ImagePoints[:,0],-ImagePoints[:,1])
ax1.plot([0,2000,2000,0,0],[0,0,-2000,-2000,0])
ax1.axis("equal")
plt.show()

#####################################################################################################################################
noise = 1*np.random.random((ImagePoints.shape))
ImagePoints = ImagePoints + noise

print(DLT.unittest(Image1, ControlPoints, ImagePoints))

####################################################################################################################################

'''initialValues=np.array([1536+20*np.random.random(),5+3*np.random.random(),4+3*np.random.random(),15+3*np.random.random(),(4+np.random.random())*np.pi/180,(5+np.random.random())*np.pi/180,(3+np.random.random())*np.pi/180,994+20*np.random.random(),1012+20*np.random.random(),0,0])
Result = camera1.Calibration(ImagePoints,ControlPoints,initialValues,Image1)
MM.PrintMatrix(Result)'''

x=9