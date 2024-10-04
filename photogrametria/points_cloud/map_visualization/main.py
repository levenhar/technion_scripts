import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

xcorner = 1000
ycorner = 1000
cellsize =50
rows = 160
columns = 353

HightGrid = pd.read_csv("HightGrid.csv", skiprows=6, header=None,usecols=list(range(160)),sep = " ")
HightGrid = HightGrid.to_numpy()

GrayLavel = (HightGrid - HightGrid.min())/(HightGrid - HightGrid.min()).max()
GrayLavel *= 255

plt.imshow(GrayLavel, cmap = "gray")
plt.show()


Hdx = 1/(8*cellsize)*np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Hdy = Hdx.T
dldx=cv2.filter2D(HightGrid,-1,Hdx)
dldy=cv2.filter2D(HightGrid,-1,Hdy)

Asp = np.arctan2(dldy,dldx)

######## User Input ###########################################################
Z = np.deg2rad(45)
AZg = np.deg2rad(135)
#######################################################################################

AZm = 2*np.pi - (AZg + np.pi/2)
S = np.sqrt(dldx**2+dldy**2)

HillShade = 255*(np.cos(Z)*np.cos(S)+np.sin(Z)*np.sin(S)*np.cos(AZm-Asp))


#####################################################################################
SlopeMap = (S - S.min()) / (S - S.min()).max()
SlopeMap *= 255
####################################################################################
AspectMap = np.ones((Asp.shape[0],Asp.shape[1],3))
AspectMap[:,:,0] = Asp

ONESS = np.ones(Asp.shape)
AspectMap = (180/np.pi*AspectMap).astype(np.uint8)
AspectMapRGB = cv2.cvtColor(AspectMap, cv2.COLOR_HSV2BGR)


##################################################################################
SlopeAspectMap = np.copy(AspectMap)
SlopeAspectMap[:,:,1] = S*255
SlopeAspectMap = (SlopeAspectMap).astype(np.uint8)

SlopeAspectMapRGB = cv2.cvtColor(SlopeAspectMap, cv2.COLOR_HSV2BGR)

fig1 = plt.figure()
ax1 = fig1.add_subplot(141)
ax2 = fig1.add_subplot(142)
ax3 = fig1.add_subplot(143)
ax4 = fig1.add_subplot(144)

ax1.imshow(SlopeMap, cmap ="gray")
ax1.set_title("Slopes Map")

ax2.imshow(HillShade, cmap = "gray")
ax2.set_title("HillShade Map")

ax3.imshow(AspectMapRGB+30)
ax3.set_title("Aspect Map")

ax4.imshow(SlopeAspectMapRGB)
ax4.set_title("Slope - Aspect Map")

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
plt.show()

####### NEW GRID #########################################################################


Hdx = 1/(2*cellsize)*np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
Hdy = Hdx.T
Hdxy = 1/(4*cellsize)*np.array([[1,0,-1], [0,0,0], [1, 0, -1]])
dldx=cv2.filter2D(HightGrid,-1,Hdx)
dldy=cv2.filter2D(HightGrid,-1,Hdy)
dldxy=cv2.filter2D(HightGrid,-1,Hdxy)

XX1 = np.array([[1,0,0,0],[0,1,0,0],[-3,-2,3,-1],[2,1,-2,1]])


######## User Input - New grid parameters ###############################################
LeftButomX = 1125
LeftButomY = 1125
NewCellSize = 35
NumRows = 450
NumColumn = 220
#################################################################################

X, Y = np.meshgrid(range(LeftButomX,LeftButomX+NewCellSize*NumColumn,NewCellSize),range(LeftButomY+NewCellSize*(NumRows),LeftButomY-1,-NewCellSize))


##### Indexes in old map #############################################
J = ((X - xcorner)/cellsize)
I = ((Y - ycorner)/cellsize)

if J.max() > len(dldx[0]) or I.max() > len(dldx) or J.min() < 0 or I.min() < 0:
    raise Exception("The new grid boundary is out of the original map")



##### Indexes in new map #############################################
JJ = ((X - LeftButomX)/NewCellSize)
II = ((Y - LeftButomY)/NewCellSize)

NewGRIDHigt = np.zeros(J.shape)
for indexi, i in enumerate(I[:,0]):
    for indexj, j in enumerate(J[0,:]):
        H00 = np.array([[HightGrid[int(i)-1,int(j)-1],dldx[int(i)-1,int(j)-1]],[dldy[int(i)-1,int(j)-1],dldxy[int(i)-1,int(j)-1]]])
        H01 = np.array([[HightGrid[int(i)-1,int(j)+1],dldx[int(i)-1,int(j)+1]],[dldy[int(i)-1,int(j)+1],dldxy[int(i)-1,int(j)+1]]])
        H11 = np.array([[HightGrid[int(i)+1,int(j)-1],dldx[int(i)+1,int(j)-1]],[dldy[int(i)+1,int(j)-1],dldxy[int(i)+1,int(j)-1]]])
        H10 = np.array([[HightGrid[int(i)+1,int(j)+1],dldx[int(i)+1,int(j)+1]],[dldy[int(i)+1,int(j)+1],dldxy[int(i)+1,int(j)+1]]])

        H1 = np.concatenate((H00,H01))
        H2 = np.concatenate((H10,H11))
        H = np.concatenate((H1,H2),axis=1)

        A = XX1@H@XX1.T

        x = j - int(j)
        y = i - int(i)

        XX = np.array([[1, x, x**2, x**3]])
        YY = np.array([[1, y, y**2, y**3]])

        h = XX@A@YY.T

        NewGRIDHigt[int(II[indexi,indexj]),int(JJ[indexi,indexj])] = h



fig2 = plt.figure()
ax21 = fig2.add_subplot(121)
ax22 = fig2.add_subplot(122)

NewGRIDHigtMap = (NewGRIDHigt - NewGRIDHigt.min())/(NewGRIDHigt - NewGRIDHigt.min()).max()
NewGRIDHigtMap *= 255


ax21.imshow(NewGRIDHigtMap, cmap = "gray")
ax21.set_title("New Height Grid")
ax22.imshow(GrayLavel, cmap = "gray")
ax22.set_title("Original Height Grid")

ax21.set_xticks([])
ax21.set_yticks([])
ax22.set_xticks([])
ax22.set_yticks([])
plt.show()

# Example strings
string1 = f"ncols {NumColumn}"
string2 = f"nrows {NumRows}"
string3 = f"xllcorner {LeftButomX}"
string4 = f"yllcorner {LeftButomY}"
string5 = f"cellsize {NewCellSize}"
string6 = f"NODATA_value  -9999"

# Save the array to a text file
filename = "NewGrid.txt"

# Open the file in write mode
with open(filename, 'w') as file:
    # Write the strings to the file
    file.write(string1 + '\n')
    file.write(string2 + '\n')
    file.write(string3 + '\n')
    file.write(string4 + '\n')
    file.write(string5 + '\n')
    file.write(string6 + '\n')

    # Write the array to the file
    np.savetxt(file, NewGRIDHigtMap, fmt='%d')

print("Array data saved to", filename)

x=9
