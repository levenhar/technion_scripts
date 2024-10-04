from Reader import Reader as R
import Camera as Cam
from SingleImage import SingleImage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import MatrixMethods as MM
from ImagePair import ImagePair



#Lab 5:
#######################################

CamFile=R.ReadCamFile("rc30.cam")
FidushelsImage74=R.Readtxtfile("fiducialsImg3574.txt")
FidushelsImage75=R.Readtxtfile("fiducialsImg3575.txt")

f=CamFile["f"]
xp=CamFile["xp"]
yp=CamFile["yp"]
fiducials=CamFile["fiducials"]
k0=CamFile["k0"]
k1=CamFile["k1"]
k2=CamFile["k2"]
k3=CamFile["k3"]
p1 =CamFile["p1"]
p2 =CamFile["p2"]
p3 =CamFile["p3"]
p4 =CamFile["p4"]

Cam1=Cam.Camera(f,np.array([xp,yp]),np.array([k0,k1,k2,k3]),np.array([p1,p2,p3,p4]),fiducials)

Image1=SingleImage(Cam1)  #SingleImage object for 74 Image
Image2=SingleImage(Cam1)  #SingleImage object for 75 Image


#calculate Oriantation
Image1.ComputeInnerOrientation(FidushelsImage74,-1)
Image2.ComputeInnerOrientation(FidushelsImage75,-1)

images, grdPnts, imgPnts = R.photoModXMLReader("Lab5 [1].x-points")

imgPnts74=np.delete(imgPnts,[0,2,4,6,8,10,15,16],0)
imgP74= imgPnts74[:, 1:3]

imgPnts75=np.delete(imgPnts,[1,3,5,7,9,11,12,13,14],0)
imgP75= imgPnts75[:, 1:3]


#### convert all the coordinate to float

imgP1 = [float(numeric_string) for numeric_string in imgP74[:, 0]]
imgP2 = [float(numeric_string) for numeric_string in imgP74[:, 1]]
imgP74=np.array([imgP1, imgP2]).T

imgP1 = [float(numeric_string) for numeric_string in imgP75[:, 0]]
imgP2 = [float(numeric_string) for numeric_string in imgP75[:, 1]]
imgP75=np.array([imgP1, imgP2]).T

grdP74= grdPnts[0:9, 2:]
grdP75=grdPnts[0:17, 2:]
grdP75=np.delete(grdP75,[6,7,8,9,10,11,12,13,15],0)

grdP1 = [float(numeric_string) for numeric_string in grdP74[:, 0]]
grdP2 = [float(numeric_string) for numeric_string in grdP74[:, 1]]
grdP3 = [float(numeric_string) for numeric_string in grdP74[:, 2]]
grdP74=np.array([grdP1, grdP2, grdP3]).T

grdP1 = [float(numeric_string) for numeric_string in grdP75[:, 0]]
grdP2 = [float(numeric_string) for numeric_string in grdP75[:, 1]]
grdP3 = [float(numeric_string) for numeric_string in grdP75[:, 2]]
grdP75=np.array([grdP1, grdP2, grdP3]).T


########calculate Exterior Orientation

####section A - All points
Image1.ComputeExteriorOrientation(imgP74, grdP74, 0.001)
Image2.ComputeExteriorOrientation(imgP75, grdP75, 0.001)
#####################################################################

###Lab 6

__, __, imgPnts6 = R.photoModXMLReader("CheckPoints.x-points")
imgP74=np.zeros((4,2))
imgP75=np.zeros((4,2))
c=0

'''points=pd.read_csv("PointsImagePaint.csv",header=None)
imgPnts6=points.to_numpy()'''

for i in range(0,len(imgPnts6),2):
    imgP75[c,0]=imgPnts6[i,1]
    imgP75[c,1]=imgPnts6[i,2]
    imgP74[c, 0] = imgPnts6[i+1, 1]
    imgP74[c, 1] = imgPnts6[i+1, 2]
    c+=1

imgP1 = [float(numeric_string) for numeric_string in imgP74[:, 0]]
imgP2 = [float(numeric_string) for numeric_string in imgP74[:, 1]]
imgP74 = np.array([imgP1, imgP2]).T

imgP1 = [float(numeric_string) for numeric_string in imgP75[:, 0]]
imgP2 = [float(numeric_string) for numeric_string in imgP75[:, 1]]
imgP75=np.array([imgP1, imgP2]).T


### convert the points from the image to the camera system
CamPoints74=Image1.ImageToCamera(imgP74)
CamPoints75=Image2.ImageToCamera(imgP75)



Table74=np.concatenate((imgP74,CamPoints74),axis=1)
MM.PrintMatrix(Table74,["x image [pix]","y image [pix]", "x camera [mm]", "y camera [mm]"])

Table75=np.concatenate((imgP75,CamPoints75),axis=1)
MM.PrintMatrix(Table75,["x image [pix]","y image [pix]", "x camera [mm]", "y camera [mm]"])

### section B - function
SterioModel=ImagePair(Image1,Image2)


#print(SterioModel.ImagesToGround(CamPoints74,CamPoints75,"geometric"))
GrdPoints=(SterioModel.ImagesToGround(CamPoints74,CamPoints75,"vector"))
print(GrdPoints)

#conver GrdPoints from dict to array
G=np.zeros((len(GrdPoints)//2,3))
for i in range(len(GrdPoints)//2):
    G[i,:]=GrdPoints[f"P{i}"]

MM.PrintMatrix(Image1.GroundToImage(G))
MM.PrintMatrix(Image2.GroundToImage(G))