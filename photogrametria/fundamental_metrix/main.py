import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MatrixMethods
from Camera import Camera
from SingleImage import SingleImage
import cv2
import os
from scipy.interpolate import RegularGridInterpolator

def ForwardCrossing(K,Img1Point, Img2Point, R1, R2, O1, O2):
    V1 = (R1@np.linalg.inv(K)@Img1Point)
    V2 = (R2@np.linalg.inv(K)@Img2Point)

    V1 = V1 / np.linalg.norm(V1)
    V2 = V2 / np.linalg.norm(V2)


    A = np.array([[(V1.T@V1)[0,0], (-V2.T@V1)[0,0]],[(-V2.T@V1)[0,0], (V2.T@V2)[0,0]]])
    b = np.array([[((O2-O1).T@V1)[0,0]],[(-(O2-O1).T@V2)[0,0]]])

    s12 = np.linalg.solve(A,b)

    P1  = O1 + s12[0]*V1
    P2 = O2 + s12[1]*V2

    #print(P1-P2)
    return (P1+P2)/2, s12



def read_images_from_folder(folder_path, isblack):
    images = []
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPEG')):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            if isblack:
                image = cv2.imread(image_path, 0)
            else:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is not None:
                images.append(image)
    return images


def FindFmatrix(Pointsimage1,Pointsimage2, Nmatrix):
    P1 = np.concatenate((Pointsimage1, np.ones((len(Pointsimage1),1))), axis = 1).T
    P2 = np.concatenate((Pointsimage2, np.ones((len(Pointsimage2),1))), axis = 1).T
    Pointsimage1 = (Nmatrix@P1).T
    Pointsimage2 = (Nmatrix@P2).T

    c1 = (Pointsimage2[:,0]*Pointsimage1[:,0]).reshape((len(Pointsimage2),1))
    c2 = (Pointsimage2[:,0]*Pointsimage1[:,1]).reshape((len(Pointsimage2),1))
    c3 = (Pointsimage2[:,0]).reshape((len(Pointsimage2),1))
    c4 = (Pointsimage1[:,0]*Pointsimage2[:,1]).reshape((len(Pointsimage2),1))
    c5 = (Pointsimage1[:,1]*Pointsimage2[:,1]).reshape((len(Pointsimage2),1))
    c6 = (Pointsimage2[:,1]).reshape((len(Pointsimage2),1))
    c7 = (Pointsimage1[:,0]).reshape((len(Pointsimage2),1))
    c8 = (Pointsimage1[:, 1]).reshape((len(Pointsimage2), 1))
    c9 = np.ones((len(Pointsimage2),1))

    A = np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8,c9), axis=1)
    U, s, V = np.linalg.svd(A)

    null_space = V[-1]
    F = null_space.reshape((3,3))
    U, s, V = np.linalg.svd(F)
    s[-1] = 0

    F = U@np.diag(s)@V
    F = Nmatrix.T@F@Nmatrix
    return F

def plot_line(Line, ax, img):
    # Generate x-coordinates for the line
    x = np.linspace(0, img.shape[1], 100)
    # Calculate corresponding y-coordinates using the line equation
    y = (-Line[0,0] * x - Line[0,2]) / Line[0,1]
    # Plot the line
    ax.plot(x, y, color='m', linewidth=0.8)

# Specify the folder path and the number of images to read
folder_path = "CalibrateImages"  # Replace with the actual folder path

# Read images from the folder
imagesK = read_images_from_folder(folder_path, True)

Cam1 = Camera(None, None, [], [], [],[])
K = Cam1.ChessCalibration(imagesK)

folder_path = "Images"
images = read_images_from_folder(folder_path, False)

##### Reading the images
Img1 = cv2.imread("IMG_3083.JPEG")
Img1 = cv2.cvtColor(Img1, cv2.COLOR_BGR2RGB)

Img2 = cv2.imread("IMG_3082.JPEG")
Img2 = cv2.cvtColor(Img2, cv2.COLOR_BGR2RGB)


DgimutPix4D = pd.read_csv("Lab8Dgimut.csv").to_numpy()

Nmatrix = np.array([[2/Img1.shape[1],0,-1], [0 , 2/Img1.shape[0], -1], [0, 0, 1]])

pts1 = DgimutPix4D[:,1:3].astype(np.float)
pts2 = DgimutPix4D[:,3:].astype(np.float)

#plot all the Points ###########################################################
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
color = np.random.randint(0,255,len(pts1))
ax1.imshow(Img1)
ax2.imshow(Img2)

ax1.scatter(pts1[:, 0], pts1[:, 1], c=color)
ax2.scatter(pts2[:, 0], pts2[:, 1], c=color)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()


#### Calculate F matrix
for i in range(250):
    #n = np.random.randint(0,len(pts1),8)
    n = np.random.choice(range(len(pts1)), size=8, replace=False)
    pts1I = pts1[n,:]
    pts2I = pts2[n,:]
    F = FindFmatrix(pts1I,pts2I,Nmatrix)
    goodPoints = []
    for k in range(len(pts1)):
        P1 = np.concatenate((pts1[k,:],np.array([1]))).reshape((3,1))
        P2 = np.concatenate((pts2[k,:],np.array([1]))).reshape((3,1))
        if np.any(F == None):
            continue
        Line = P2.T@F
        x = (abs(Line@P1) / np.sqrt(Line[:,:2]@Line[:,:2].T))[0,0]
        if x<=3:
            goodPoints.append(k)

    print(len(goodPoints)/len(pts1))
    if len(goodPoints)/len(pts1) > 0.70:
        print(f"we stop at iteration {i+1}")
        break
else:
    #print(f"after 5 iteration we could not find the solution")
    raise Exception("after 250 iteration we could not find the solution")


#plot the sift results ###########################################################
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
color = np.random.randint(0,255,len(goodPoints))
ax1.imshow(Img1)
ax2.imshow(Img2)

ax1.scatter(pts1[goodPoints, 0], pts1[goodPoints, 1], c=color)
ax2.scatter(pts2[goodPoints, 0], pts2[goodPoints, 1], c=color)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()
####################################################################################

#Adjust F without outliers
F = FindFmatrix(pts1[goodPoints, :],pts2[goodPoints, :],Nmatrix)

MatrixMethods.PrintMatrix(F)
############# Disply the apipolar line and the apipol##########################################
fig1 = plt.figure()
ax11 = fig1.add_subplot(121)
ax21 = fig1.add_subplot(122)
ax11.imshow(Img1)
ax21.imshow(Img2)

ax11.set_xticks([])
ax11.set_yticks([])
ax21.set_xticks([])
ax21.set_yticks([])

for i in goodPoints:
    P1 = np.concatenate((pts1[i, :], np.array([1]))).reshape((3, 1))
    Line =(F @ P1).reshape((1,3))
    plot_line(Line,ax21,Img2)
ax11.scatter(pts1[goodPoints, 0], pts1[goodPoints, 1], c=color)

# U, s, V = np.linalg.svd(F)
# et = V[-1]
# et = et/et[-1]
# ax21.scatter(et[0], et[1])
#
# e = U[-1]
# e = e/e[-1]
# ax21.scatter(e[0], e[1])

plt.show()
###############################################################################
################Finding E matrix ##############################################

E = K.T@F@K
U, s, V = np.linalg.svd(E)
Snew = np.eye(3)
Snew[-1,-1] = 0
ENormal = U@Snew@V

##### Calculate the Relative Orintation ################

_, R, b, _ = cv2.recoverPose(ENormal, pts1[goodPoints, :], pts2[goodPoints, :])

# W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
#
# t1 = U[:,-1].reshape(3,1)
# t2 = -U[:,-1].reshape(3,1)
#
# R1 = U@W@V.T
# R2 = U@W.T@V.T
#
Img1Points = np.concatenate((pts1[goodPoints,:], np.ones((len(goodPoints),1))),axis=1)
Img2Points = np.concatenate((pts2[goodPoints,:], np.ones((len(goodPoints),1))),axis=1)
#
# X, k12 = ForwardCrossing(K, Img1Points[0,:].reshape(3,1),Img2Points[0,:].reshape(3,1), np.eye(3), R1, np.zeros((3,1)), t1)
# if any(k12<0):
#     X, k12 = ForwardCrossing(K, Img1Points[0,:].reshape(3,1),Img2Points[0,:].reshape(3,1), np.eye(3), R1, np.zeros((3,1)), t2)
#     if any(k12 < 0):
#         X, k12 = ForwardCrossing(K, Img1Points[0, :].reshape(3, 1), Img2Points[0, :].reshape(3, 1), np.eye(3), R2,
#                                  np.zeros((3, 1)), t2)
#         if any(k12 < 0):
#             X, k12 = ForwardCrossing(K, Img1Points[0, :].reshape(3, 1), Img2Points[0, :].reshape(3, 1), np.eye(3), R2,
#                                      np.zeros((3, 1)), t1)
#             if any(k12 < 0):
#                 print("Error")
#             else:
#                 print("R2, t1")
#                 R = R2
#                 b = t1
#         else:
#             print("R2, t2")
#             R = R2
#             b = t2
#     else:
#         print("R1, t2")
#         R = R1
#         b = t2
# else:
#     print("R1, t1")
#     R = R1
#     b = t1


modelPoints = []
for i in range(len(goodPoints)):
    P1 = Img1Points[i, :].reshape(3, 1)
    P2 = Img2Points[i, :].reshape(3, 1)
    X, k12 = ForwardCrossing(K, P1, P2, np.eye(3), R,np.zeros((3, 1)), b)
    modelPoints.append(X.T)

modelPoints = np.concatenate(modelPoints)

##### ColoringTheData ################################################################
colors = np.zeros((len(goodPoints), 3))
x_coords = np.linspace(0, len(Img1[0])-1, len(Img1[0]))
y_coords = np.linspace(0, len(Img1)-1, len(Img1))
interpolator = RegularGridInterpolator((y_coords, x_coords), Img1)

for i in range(len(goodPoints)):
    color = interpolator((pts1[goodPoints[i],1], pts1[goodPoints[i],0]))
    colors[i,:] = color.astype(int)

fig3D = plt.figure()
ax3D = fig3D.add_subplot(111, projection ="3d")
ax3D.scatter(modelPoints[:,0],modelPoints[:,1],modelPoints[:,2], marker=".", c = colors/255)



plt.show()
x=0
