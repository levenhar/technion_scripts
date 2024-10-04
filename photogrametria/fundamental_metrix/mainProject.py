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
    V1 = (R1@(Img1Point-kk))
    V2 = (R2@(Img2Point-kk))

    V1 = V1 / np.linalg.norm(V1)
    V2 = V2 / np.linalg.norm(V2)


    A = np.array([[(V1.T@V1)[0,0], (-V2.T@V1)[0,0]],[(-V2.T@V1)[0,0], (V2.T@V2)[0,0]]])
    b = np.array([[((O2-O1).T@V1)[0,0]],[(-(O2-O1).T@V2)[0,0]]])

    s12 = np.linalg.solve(A,b)

    P1  = O1 + s12[0]*V1
    P2 = O2 + s12[1]*V2

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
                #image = image.T
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

xp = K[0,2]
yp = K[1,2]
f = (K[0,0]+K[1,1])/2

kk = np.array([[xp],[yp],[f]])

folder_path = "Images"
images = read_images_from_folder(folder_path, False)

fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax1.imshow(images[0])
ax2.imshow(images[1])
ax3.imshow(images[2])
ax4.imshow(images[3])
ax5.imshow(images[4])
ax6.imshow(images[5])

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])
plt.show()


Fs = []
Es = []
Rs = []
bs = []
imgPoints1 = []
imgPoints2 = []


for i in range(len(images)-2):
    Img1 = images[i]
    Img2 = images[i+1]

    gray1 = cv2.cvtColor(Img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(Img2, cv2.COLOR_RGB2GRAY)

    gray1_g= cv2.GaussianBlur(gray1,(5, 5),1)
    gray2_g = cv2.GaussianBlur(gray2,(5, 5),1)

    # keypoints detection
    sift = cv2.ORB_create()
    kp1, des1 = sift.detectAndCompute(gray1_g, None)
    kp2, des2 = sift.detectAndCompute(gray2_g, None)

    # keypoints matching
    indexParams = dict(algorithm=0, trees=5)
    searchParams = dict(checks=50)
    #
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    bf = cv2.BFMatcher()
    #
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    #
    # filter out “far” homological points
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.85*n.distance:  # change 0.85 to the threshold needed
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    #
    print(f'Number of matches: {len(good)}')
    pts1 = np.array(np.array(pts1))
    pts2 = np.array(np.array(pts2))

    Nmatrix = np.array([[2/Img1.shape[1],0,-1], [0 , 2/Img1.shape[0], -1], [0, 0, 1]])

    # ############# display sift result ##########################################
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    #
    # ax1.imshow(Img1)
    # ax2.imshow(Img2)
    #
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    #
    # ax1.scatter(pts1[:,0],pts1[:,1])
    # ax2.scatter(pts2[:, 0], pts2[:, 1])
    #
    # plt.show()


    #### Calculate F matrix
    IndexesList = []
    SuccessRates = []
    for i in range(272):
        #n = np.random.randint(0,len(pts1),8)
        n = np.random.choice(range(len(pts1)), size=8, replace=False)
        pts1I = pts1[n,:]
        pts2I = pts2[n,:]
        F = FindFmatrix(pts1I,pts2I,Nmatrix)
        if np.any(F == None):
            continue
        goodPoints = []
        for k in range(len(pts1)):
            P1 = np.concatenate((pts1[k,:],np.array([1]))).reshape((3,1))
            P2 = np.concatenate((pts2[k,:],np.array([1]))).reshape((3,1))
            Line = P2.T@F
            x = (abs(Line@P1) / np.sqrt(Line[:,:2]@Line[:,:2].T))[0,0]
            if x<=5:
                goodPoints.append(k)
        IndexesList.append(np.array(goodPoints))
        SuccessRates.append(len(goodPoints) / len(pts1))
    SuccessRates = np.array(SuccessRates)
    maxIndex = np.where(SuccessRates==max(SuccessRates))
    goodPoints = IndexesList[maxIndex[0][0]]
    print(max(SuccessRates))

    ####################################################################################


    ### displt ransac result
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(Img1)
    ax2.imshow(Img2)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.scatter(pts1[:, 0], pts1[:, 1], c="b")
    ax2.scatter(pts2[:, 0], pts2[:, 1], c="b")

    ax1.scatter(pts1[goodPoints, 0], pts1[goodPoints, 1], c="m")
    ax2.scatter(pts2[goodPoints, 0], pts2[goodPoints, 1], c="m")

    plt.show()

    ###################################################################################
    #Adjust F without outliers
    F = FindFmatrix(pts1[goodPoints, :],pts2[goodPoints, :],Nmatrix)


    ###############################################################################
    ################Finding E matrix #############################################

    E = K.T@F@K
    MatrixMethods.PrintMatrix(E)
    U, s, V = np.linalg.svd(E)
    Snew = np.eye(3)
    Snew[-1,-1] = 0
    ENormal = U@Snew@V

    ##### Calculate the Relative Orintation ################

    U, _, V = np.linalg.svd(ENormal)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    t1 = U[:,-1].reshape(3,1)
    t2 = -U[:,-1].reshape(3,1)

    R1 = U@W@V
    R2 = U@W.T@V
    # #
    Img1Points = np.concatenate((pts1[goodPoints,:], np.ones((len(goodPoints),1))),axis=1)
    Img2Points = np.concatenate((pts2[goodPoints,:], np.ones((len(goodPoints),1))),axis=1)
    #
    #
    X, k12 = ForwardCrossing(K, Img1Points[0,:].reshape(3,1),Img2Points[0,:].reshape(3,1), np.eye(3), R1, np.zeros((3,1)), t1)
    if any(k12<0):
        X, k12 = ForwardCrossing(K, Img1Points[0,:].reshape(3,1),Img2Points[0,:].reshape(3,1), np.eye(3), R1, np.zeros((3,1)), t2)
        if any(k12 < 0):
            X, k12 = ForwardCrossing(K, Img1Points[0, :].reshape(3, 1), Img2Points[0, :].reshape(3, 1), np.eye(3), R2,
                                     np.zeros((3, 1)), t2)
            if any(k12 < 0):
                X, k12 = ForwardCrossing(K, Img1Points[0, :].reshape(3, 1), Img2Points[0, :].reshape(3, 1), np.eye(3), R2,
                                         np.zeros((3, 1)), t1)
                if any(k12 < 0):
                    raise Exception("could not find correct R and b")
                else:
                    print("R2, t1")
                    R = R2
                    b = t1
            else:
                print("R2, t2")
                R = R2
                b = t2
        else:
            print("R1, t2")
            R = R1
            b = t2
    else:
        print("R1, t1")
        R = R1
        b = t1

    _, R, b, _ = cv2.recoverPose(ENormal, pts1[goodPoints, :], pts2[goodPoints, :])

    Fs.append(F)
    Es.append(ENormal)
    Rs.append(R)
    bs.append(b)
    imgPoints1.append(Img1Points)
    imgPoints2.append(Img2Points)


for j in range(len(Rs)-2,-1,-1):
    for k in range(len(Rs)-1,j,-1):
        Rs[k] = Rs[j]@Rs[k]


##### Linkage the models ##########################################################################

point4Scale = pd.read_csv("Dgimot4Scales.csv", header=None)
point4Scale = point4Scale.to_numpy()

Point1 = point4Scale[:6,:]
Point2 = point4Scale[6:,:]

v1 = Point1[0,:].reshape((3,1)) - kk
Vs = [v1.T/np.linalg.norm(v1)]

## ajusnent for scale values ####
for inx in range(len(Point1)-1):
    V = Rs[inx]@(Point1[inx+1,:].reshape((3,1))-kk)
    V = V/np.linalg.norm(V)
    Vs.append(V.T)

Vs = np.concatenate(Vs)
Rs.insert(0,np.eye(3))
bs.insert(0,np.zeros((3,1)))

L=np.zeros((3*len(Vs),1))
A1 = np.zeros((3*len(Vs),3))
c=0
I = np.eye(3)
for i in range(0,len(L),3):
    vi = Vs[c,:].reshape((3,1))
    A1[i:i+3,:] = I - vi@vi.T
    L[i:i + 3, :] = A1[i:i+3,:]@bs[1].reshape((3,1))
    c+=1

L[0:3,0] = np.array([0,0,0])

A2 = np.zeros((3*len(Vs),len(Vs)-2))

for i in range(6,len(A2),3):
    for j in range(0,i//3-1):
        a = -A1[i:i+3,:]@Rs[j+1]@bs[j+2].reshape((3,1))
        A2[i:i+3,j] = a.reshape(-1,)

A = np.concatenate((A1,A2),axis=1)

X1 = np.linalg.solve(A.T@A,A.T@L)
########################################

v1 = Point2[0,:].reshape((3,1)) - kk
Vs = [v1.T/np.linalg.norm(v1)]

## ajusnent for scale values ####
for inx in range(len(Point2)-1):
    V = Rs[inx]@(Point2[inx+1,:].reshape((3,1))-kk)
    V = V/np.linalg.norm(V)
    Vs.append(V.T)

Vs = np.concatenate(Vs)

L=np.zeros((3*len(Vs),1))
A1 = np.zeros((3*len(Vs),3))
c=0
I = np.eye(3)
for i in range(0,len(L),3):
    vi = Vs[c,:].reshape((3,1))
    A1[i:i+3,:] = I - vi@vi.T
    L[i:i + 3, :] = A1[i:i+3,:]@bs[1].reshape((3,1))
    c+=1

L[0:3,0] = np.array([0,0,0])

A2 = np.zeros((3*len(Vs),len(Vs)-2))

for i in range(6,len(A2),3):
    for j in range(0,i//3-1):
        a = -A1[i:i+3,:]@Rs[j+1]@bs[j+2].reshape((3,1))
        A2[i:i+3,j] = a.reshape(-1,)

A11 = np.concatenate((A1,A2),axis=1)

X2 = np.linalg.solve(A11.T@A,A11.T@L)
Scales = (X1[3:] + X2[3:])/2

MatrixMethods.PrintMatrix(Scales)

# calculate the position of each image ##########
Os = bs[0:2]

for i in range(2,len(bs)):
    o = Os[i-1] + Scales[i-2]*Rs[i-1]@bs[i]
    Os.append(o)


########## for interpulation - coloring the points
x_coords = np.linspace(0, len(Img1[0])-1, len(Img1[0]))
y_coords = np.linspace(0, len(Img1)-1, len(Img1))
#########

ModelsPoints = []
errorcount = 0
for i in range(len(Fs)):
    modelPoints = []
    # if i==2:
    #     continue
    R1 = Rs[i]
    R2 = Rs[i + 1]
    O1 = Os[i]
    O2 = Os[i + 1]
    for p in range(len(imgPoints1[i])):
        P1 = imgPoints1[i][p, :].reshape(3, 1)
        P2 = imgPoints2[i][p, :].reshape(3, 1)


        X, k12 = ForwardCrossing(K, P1, P2, R1, R2,O1, O2)
        if any(k12)<0:
            errorcount += 1
            break
        modelPoints.append(X.T)
    print(errorcount)

    modelPoints = np.concatenate(modelPoints)

    ##### ColoringTheData ################################################################
    colors = np.zeros((len(imgPoints1[i]), 3))

    interpolator = RegularGridInterpolator((y_coords, x_coords), images[i])

    for p in range(len(imgPoints1[i])):
        color = interpolator((imgPoints1[i][p,1], imgPoints1[i][p,0]))
        colors[p,:] = color.astype(int)
    fig3D1 = plt.figure()
    ax3D1 = fig3D1.add_subplot(111, projection="3d")
    ax3D1.scatter(modelPoints[:, 0], modelPoints[:, 1], modelPoints[:, 2], marker=".", c=colors / 255)
    plt.show()
    if i != 2 and i !=3:
        ModelsPoints.append(np.concatenate((modelPoints,colors),axis=1))

ModelsPoints = np.concatenate(ModelsPoints)

### disply the points on the images
fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax1.imshow(images[0])
ax2.imshow(images[1])
ax3.imshow(images[2])
ax4.imshow(images[3])
ax5.imshow(images[4])
ax6.imshow(images[5])

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax4.set_xticks([])
ax4.set_yticks([])
ax5.set_xticks([])
ax5.set_yticks([])
ax6.set_xticks([])
ax6.set_yticks([])

ax1.scatter(imgPoints1[0][:,0],imgPoints1[0][:,1])
ax2.scatter(imgPoints1[1][:,0],imgPoints1[1][:,1])
ax3.scatter(imgPoints1[2][:,0],imgPoints1[2][:,1])
ax4.scatter(imgPoints1[3][:,0],imgPoints1[3][:,1])
ax5.scatter(imgPoints1[4][:,0],imgPoints1[4][:,1])
ax6.scatter(imgPoints2[4][:,0],imgPoints2[4][:,1])


ax1.scatter(point4Scale[0,0],point4Scale[0,1], c="red")
ax2.scatter(point4Scale[1,0],point4Scale[1,1], c="red")
ax3.scatter(point4Scale[2,0],point4Scale[2,1], c="red")
ax4.scatter(point4Scale[3,0],point4Scale[3,1], c="red")
ax5.scatter(point4Scale[4,0],point4Scale[4,1], c="red")
ax6.scatter(point4Scale[5,0],point4Scale[5,1], c="red")

ax1.scatter(point4Scale[6,0],point4Scale[6,1], c="red")
ax2.scatter(point4Scale[7,0],point4Scale[7,1], c="red")
ax3.scatter(point4Scale[8,0],point4Scale[8,1], c="red")
ax4.scatter(point4Scale[9,0],point4Scale[9,1], c="red")
ax5.scatter(point4Scale[10,0],point4Scale[10,1], c="red")
ax6.scatter(point4Scale[11,0],point4Scale[11,1], c="red")
plt.show()

###########################################################



#### display the result ##################################
fig3D = plt.figure()
ax3D = fig3D.add_subplot(111, projection ="3d")
ax3D.scatter(ModelsPoints[:,0],ModelsPoints[:,1],ModelsPoints[:,2], marker=".", c = ModelsPoints[:,3:]/255)

plt.show()


########## Absolute Orintation######################################################################

point4Orintation = pd.read_csv("Points4Orintation.csv", header=None)
point4Orintation = point4Orintation.to_numpy()

PImage1 = point4Orintation[:,:3]
PImage2 = point4Orintation[:,3:6]
PReal = point4Orintation[:,6:]

############
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(images[1])
ax2.imshow(images[2])

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

ax1.scatter(PImage1[:,0],PImage1[:,1], c="m")
ax2.scatter(PImage2[:,0],PImage2[:,1], c="m")
plt.show()

###########################################################################################
k=1

O1 = Os[k]
R1 = Rs[k]
O2 = Os[k+1]
R2 = Rs[k+1]

Points = []
for i  in range(3):
    p, k12 = ForwardCrossing(K, PImage1[i,:].reshape((3,1)), PImage2[i,:].reshape((3,1)), R1, R2,O1, O2)
    Points.append(p)

Points = np.concatenate(Points,axis=1).T
#calculate the points in the model


PavgM = np.average(Points, axis=0).reshape(1,-1)
PointsM = Points - PavgM

PavgR = np.average(PReal, axis=0).reshape(1,-1)
PointsR = PReal - PavgR

SR = np.sum(np.linalg.norm(PointsR,axis=1).reshape(-1,1)**2)
SM = np.sum(np.linalg.norm(PointsM,axis=1).reshape(-1,1)**2)

L = np.sqrt(SR/SM)

print(L)

Rtag = np.zeros((3,3))
for i in range(3):
    r = PointsM[i,:].reshape(-1,1)@PointsR[i,:].reshape(1,-1)
    Rtag += r

U, W, V = np.linalg.svd(Rtag)
R = V.T@U.T

MatrixMethods.PrintMatrix(R)

t = PavgR.T + L*R.T@PavgM.T

MatrixMethods.PrintMatrix(t)


RealPoints = (t + L*R.T@ModelsPoints[:,:3].T).T

#### display the result ##################################
fig3D = plt.figure()
ax3D = fig3D.add_subplot(111, projection ="3d")
ax3D.scatter(RealPoints[:,0],RealPoints[:,1],RealPoints[:,2], marker=".", c = ModelsPoints[:,3:]/255)

plt.show()
x=0

