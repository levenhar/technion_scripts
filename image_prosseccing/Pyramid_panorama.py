import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd


class peramidGL5():
    def __init__(self,img,isbin):
        '''
        this class convert an image to its Gauss and laplace pyramids
        each pyramid contine 5 levels
        in each level we calculate the pyramids for each band separately and than merge all to one image
        :param img: the src Image
        :type img: np.array (nxmx3) or (nxm)
        :param isbin: mark if the image is binary (nxm)
        :type isbin: bool
        '''
        self.level0 = np.copy(img) #Zero level is the original image
        self.level0 = (self.level0-self.level0.min())/((self.level0-self.level0.min()).max())  #normalized the level between 0 and 1
        #---------------Level 1------------------------------------------
        R1 = Gaussian_and_Laplas_Pyramid(self.level0[:, :, 0],isbin)
        G1 = Gaussian_and_Laplas_Pyramid(self.level0[:, :, 1],isbin)
        B1 = Gaussian_and_Laplas_Pyramid(self.level0[:, :, 2],isbin)
        self.level1 = cv2.merge((R1, G1, B1))
        self.level1 = (self.level1 - self.level1.min()) / ((self.level1 - self.level1.min()).max())#normalized the level between 0 and 1
        # ---------------Level 2------------------------------------------
        R2 = Gaussian_and_Laplas_Pyramid(self.level1[:, :, 0],isbin)
        G2 = Gaussian_and_Laplas_Pyramid(self.level1[:, :, 1],isbin)
        B2 = Gaussian_and_Laplas_Pyramid(self.level1[:, :, 2],isbin)
        self.level2 = cv2.merge((R2, G2, B2))
        self.level2 = (self.level2 - self.level2.min()) / ((self.level2 - self.level2.min()).max())#normalized the level between 0 and 1
        # ---------------Level 3------------------------------------------
        R3 = Gaussian_and_Laplas_Pyramid(self.level2[:, :, 0],isbin)
        G3 = Gaussian_and_Laplas_Pyramid(self.level2[:, :, 1],isbin)
        B3 = Gaussian_and_Laplas_Pyramid(self.level2[:, :, 2],isbin)
        self.level3 = cv2.merge((R3, G3, B3))
        self.level3 = (self.level3 - self.level3.min()) / ((self.level3 - self.level3.min()).max())#normalized the level between 0 and 1
        # ---------------Level 4------------------------------------------
        R4 = Gaussian_and_Laplas_Pyramid(self.level3[:, :, 0],isbin)
        G4 = Gaussian_and_Laplas_Pyramid(self.level3[:, :, 1],isbin)
        B4 = Gaussian_and_Laplas_Pyramid(self.level3[:, :, 2],isbin)
        self.level4 = cv2.merge((R4, G4, B4))
        self.level4 = (self.level4 - self.level4.min()) / ((self.level4 - self.level4.min()).max())#normalized the level between 0 and 1
        # ---------------Level 5------------------------------------------
        R5 = Gaussian_and_Laplas_Pyramid(self.level4[:, :, 0],isbin)
        G5 = Gaussian_and_Laplas_Pyramid(self.level4[:, :, 1],isbin)
        B5 = Gaussian_and_Laplas_Pyramid(self.level4[:, :, 2],isbin)
        self.level5 = cv2.merge((R5, G5, B5))
        self.level5 = (self.level5 - self.level5.min()) / ((self.level5 - self.level5.min()).max())#normalized the level between 0 and 1

        self.level5L = self.level4 - expend(self.level5, self.level4.shape)
        self.level5L = (self.level5L - self.level5L.min()) / ((self.level5L - self.level5L.min()).max())#normalized the level between 0 and 1

        self.level4L = self.level3 - expend(self.level4,self.level3.shape)
        self.level4L = (self.level4L - self.level4L.min()) / ((self.level4L - self.level4L.min()).max())#normalized the level between 0 and 1

        self.level3L = self.level2 - expend(self.level3,self.level2.shape)
        self.level3L = (self.level3L - self.level3L.min()) / ((self.level3L - self.level3L.min()).max())#normalized the level between 0 and 1

        self.level2L = self.level1 - expend(self.level2,self.level1.shape)
        self.level2L = (self.level2L - self.level2L.min()) / ((self.level2L - self.level2L.min()).max())#normalized the level between 0 and 1

        self.level1L = self.level0 - expend(self.level1,self.level0.shape)
        self.level1L = (self.level1L - self.level1L.min()) / ((self.level1L - self.level1L.min()).max())#normalized the level between 0 and 1
    def plot(self):
        #this function plot the pyramids according to the levels
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.level0)
        ax1.set_title("level 0 - Gauss Pyramid")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.imshow(self.level1L)
        ax2.set_title("level 1 - Laplace Pyramid")
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig1, (ax3, ax4) = plt.subplots(1, 2)
        ax3.imshow(self.level1)
        ax3.set_title("level 1 - Gauss Pyramid")
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.imshow(self.level2L)
        ax4.set_title("level 2 - Laplace Pyramid")
        ax4.set_xticks([])
        ax4.set_yticks([])

        fig2, (ax5, ax6) = plt.subplots(1, 2)
        ax5.imshow(self.level2)
        ax5.set_title("level 2 - Gauss Pyramid")
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax6.imshow(self.level3L)
        ax6.set_title("level 3 - Laplace Pyramid")
        ax6.set_xticks([])
        ax6.set_yticks([])

        fig3, (ax7, ax8) = plt.subplots(1, 2)
        ax7.imshow(self.level3)
        ax7.set_title("level 3 - Gauss Pyramid")
        ax7.set_xticks([])
        ax7.set_yticks([])
        ax8.imshow(self.level4L)
        ax8.set_title("level 4 - Laplace Pyramid")
        ax8.set_xticks([])
        ax8.set_yticks([])

        fig4, (ax9, ax10) = plt.subplots(1, 2)
        ax9.imshow(self.level4)
        ax9.set_title("level 4 - Gauss Pyramid")
        ax9.set_xticks([])
        ax9.set_yticks([])
        ax10.imshow(self.level5L)
        ax10.set_title("level 5 - Laplace Pyramid")
        ax10.set_xticks([])
        ax10.set_yticks([])

class BinGausinaPeramid():
    def __init__(self,Binimg,isBin):
        #creating a Gauss pyramid for binary image.
        self.level0=np.copy(Binimg)
        self.level1 = Gaussian_and_Laplas_Pyramid(self.level0, isBin)  #True for isbin parametr
        self.level2 = Gaussian_and_Laplas_Pyramid(self.level1, isBin)
        self.level3 = Gaussian_and_Laplas_Pyramid(self.level2, isBin)
        self.level4 = Gaussian_and_Laplas_Pyramid(self.level3, isBin)
        self.level5 = Gaussian_and_Laplas_Pyramid(self.level4, isBin)

def ComputeTransformationParameter(Observation):
    '''
    this function calculate the Homografic transform parameter and return the transform matrix between two images
    :param Observation: the homologic points in the image (x', y', x, y)
    :type Observation: np.array (nx4)
    :return: transform matrix
    :rtype: np.array (3x3)
    '''
    A=np.zeros((len(Observation)*2,8))
    l=np.zeros(((len(Observation)*2,1)))
    k=0
    for i in range(0,len(Observation)):
        #create A matrix and l vector for the adjusment procces
        xit = Observation[i, 0]
        yit = Observation[i, 1]
        xi = Observation[i, 2]
        yi = Observation[i, 3]
        A[k,0:3]=np.array([xi,yi,1])
        A[k+1, 3:6] = np.array([xi, yi, 1])
        A[k,6:]=np.array([-xi*xit,-yi*xit])
        A[k+1, 6:] = np.array([-xi * yit, -yi * yit])

        l[k,0]=xit
        l[k+1,0]=yit

        k+=2

    N=A.T@A
    U=A.T@l

    X=np.linalg.inv(N)@U

    T=np.array([[X[0,0],X[1,0],X[2,0]],[X[3,0],X[4,0],X[5,0]],[X[6,0],X[7,0],1]])
    return T

def TransforCorner(Cornner,T12,T32):
    '''
    this function calculate the new corner of the tranformed images
    :param Cornner: the corner of the origin images
    :type Cornner: np.array (3x4)
    :param T12: transform matrix of image 1
    :type T12: np.array (3x3)
    :param T32: transform matrix of image 3
    :type T32: np.array (3x3)
    :return: the new corner of image 1 and 3
    :rtype: np.array (3x4)
    '''
    Cornner1T = T12 @ Cornner
    Cornner3T = T32 @ Cornner

    Cornner1T[0, :] = Cornner1T[0, :] / Cornner1T[2, :]
    Cornner1T[1, :] = Cornner1T[1, :] / Cornner1T[2, :]
    Cornner1T[2, :] = Cornner1T[2, :] / Cornner1T[2, :]

    Cornner3T[0, :] = Cornner3T[0, :] / Cornner3T[2, :]
    Cornner3T[1, :] = Cornner3T[1, :] / Cornner3T[2, :]
    Cornner3T[2, :] = Cornner3T[2, :] / Cornner3T[2, :]

    return Cornner1T,Cornner3T

def panoramaMaking(I1,I2,I3,T12,T32):
    '''
    this fanction create a panorama from 3 Images
    also the functoin crate an image of each original image ofter the transpose (for the pyramid)
    :param I1: Image 1
    :type I1: array (nxm)
    :param I2: Image 1
    :type I2: array (nxm)
    :param I3: Image 1
    :type I3: array (nxm)
    :param T12: matrix tranform 1
    :type T12: array (3x3)
    :param T32: matrix tranform 2
    :type T32: array (3x3)
    :return: Panorama, 3 transformed images and 3 binary image that show where each image placed in the panirama
    :rtype: np.array (uxv)
    '''
    Cornner=np.array([[0,I1.shape[1],I1.shape[1],0],[0,0,I1.shape[0],I1.shape[0]],[1,1,1,1]])
    Cornner1T,Cornner3T = TransforCorner(Cornner,T12,T32)

    #calculate the size of the panorama according to the corner of the images
    bottom =max(max(Cornner1T[1, :]),max(Cornner3T[1, :]),max(Cornner[1, :]))
    top = min(min(Cornner1T[1, :]),min(Cornner3T[1, :]),min(Cornner[1, :]))
    left = min(min(Cornner1T[0, :]),min(Cornner3T[0, :]),min(Cornner[0, :]))
    right = max(max(Cornner1T[0, :]),max(Cornner3T[0, :]),max(Cornner[0, :]))

    #update the transformed matrix whis the moving factor
    TT=np.array([[1,0,abs(top)],[0,1,abs(left)+1],[0,0,1]])
    A12=TT@T12
    A32=TT@T32

    #Initioal the panorama and the Image after transform
    Panorama=np.zeros((int(bottom-top)+1,int(right-left)+1))
    Im1 = np.zeros(Panorama.shape)
    Im2 = np.zeros(Panorama.shape)
    Im3 = np.zeros(Panorama.shape)

    #------------------------------------------------------
    #for ech pixel in the panorama Image, calculate the inverse transpose and find the value from the original images.
    #first check in the middle image and than in the others
    BinaryImage1=np.zeros(Panorama.shape)
    BinaryImage2=np.zeros(Panorama.shape)
    BinaryImage3=np.zeros(Panorama.shape)
    for i in range(len(Panorama)):
        for j in range(len(Panorama[0])):
            p = np.array([[j], [i], [1]])
            pt=np.linalg.inv(A12)@p
            u = pt[0, 0] / pt[2, 0]
            v = pt[1,0] / pt[2, 0]
            if 0 < u < len(I1) and 0 < v < len(I1[0]):
                try:
                    Panorama[i, j] = I1[int(v), int(u)] #crate the panorama
                    Im1[i, j] = I1[int(v), int(u)] #create the transformed image
                    BinaryImage1[i, j] = 1  #create the binary image
                except IndexError:
                    pass
            pt = np.linalg.inv(A32) @ p
            u = pt[0, 0] / pt[2, 0]
            v = pt[1,0] / pt[2, 0]
            if 0 < u < len(I3) and 0 < v < len(I3[0]):
                try:
                    Panorama[i, j] = I3[int(v), int(u)] #crate the panorama
                    Im3[i, j] = I3[int(v), int(u)]#create the transformed image
                    BinaryImage3[i, j] = 1#create the binary image
                except IndexError:
                   pass
            pt = np.linalg.inv(TT) @ p
            u = pt[0, 0] / pt[2, 0]
            v = pt[1, 0] / pt[2, 0]
            if 0 < u < len(I2[0]) and 0 < v < len(I2):
                Panorama[i, j] = I2[int(v), int(u)]  # crate the panorama
                Im2[i, j] = I2[int(v), int(u)]  # create the transformed image
                BinaryImage2[i, j] = 1  # create the binary image
    return Panorama, Im1, Im2, Im3, BinaryImage1, BinaryImage2, BinaryImage3

def Gaussian_and_Laplas_Pyramid(img,isbin):
    '''
    create one level of the Gaussian and Laplas Pyramids
    :param img: an image
    :type img: array (nxm)
    :param isbin: if the image is binary ---> don't pass a gauss blur
    :type isbin: bool
    :return: new level of the Gaussian and Laplas Pyramids
    :rtype: laplace - np.array (nxm)  ;  gauss - np.array (n/2 x m/2)
    '''
    if not isbin:
        BlurImg = cv2.GaussianBlur(img,(5,5),1) #low pas filter
    else:
        BlurImg=np.copy(img)
    NewLevel=np.zeros((BlurImg.shape[0]//2,BlurImg.shape[1]//2)) #iniational the gauss level
    #newLaplas=img-BlurImg #Lapals level

    for i in range(0,len(BlurImg),2):
        for j in range(0,len(BlurImg[0]),2):
            try:
                NewLevel[i//2,j//2]=BlurImg[i,j]
            except IndexError:
                pass
    return NewLevel

def expend(img1,destShape):
    '''
    the function expend an image and increases its size by 2 times
    :param img1: an image
    :type img1: array(nxm)
    :param destShape: the size of the new image
    :type destShape: tuple (1x3)
    :return: the expened image
    :rtype: array(2n x 2m)
    '''
    #ExpendImg=np.zeros((img.shape[0]*2,img.shape[1]*2,3))
    # each pixel duplicate for is new neighbors
    '''j=0
    for col in range((img.shape[1])):
        i=0
        for row in range(img.shape[0]):
            ExpendImg[i, j, :] = img[row,col,:]
            ExpendImg[i+1, j, :] = img[row, col, :]
            ExpendImg[i, j+1, :] = img[row, col, :]
            ExpendImg[i+1, j+1, :] = img[row, col, :]
            i+=2
        j += 2'''
    ExpendImg=cv2.resize(img1,(destShape[1],destShape[0]))
    #correct the size according to the destShape paramers
    '''if ExpendImg.shape[0] !=destShape[0]:
        NewR=np.zeros((1, ExpendImg.shape[1], 3))
        ExpendImg=np.append(ExpendImg, NewR,axis=0)
    if ExpendImg.shape[1] !=destShape[1]:
        NewC=np.zeros((ExpendImg.shape[0], 1, 3))
        ExpendImg=np.append(ExpendImg, NewC,axis=1)'''

    ExpendImg=cv2.GaussianBlur(ExpendImg, (5, 5), 1)
    return ExpendImg

def CombineImages(img1,img2,img3,bin1,bin2,bin3):
    '''
    this function combine 3 Images to 1 image (panorama)
    :param img1: image 1
    :type img1: array (nxm)
    :param img2: image 2
    :type img2: array (nxm)
    :param img3: image 3
    :type img3: array (nxm)
    :param bin1: mask of image 1
    :type bin1: array (nxm)
    :param bin2: mask of image 2
    :type bin2: array (nxm)
    :param bin3: mask of image 3
    :type bin3: array (nxm)
    :return: combined image
    :rtype: array (nxm)
    '''
    combinedImage=np.zeros((bin1.shape[0],bin1.shape[1],3))

    for i in range(bin1.shape[0]):
        for j in range(bin1.shape[1]):
            if bin2[i,j] != 0:
                combinedImage[i, j, :] = img2[i, j, :]
            if bin1[i,j] !=0:
                combinedImage[i, j, :] = combinedImage[i, j, :]*(bin2[i,j]) + (1-bin2[i,j])*img1[i, j, :]
            if bin3[i, j] != 0:
                combinedImage[i, j, :] = combinedImage[i, j, :]*(1-bin3[i,j]) + (bin3[i,j])*img3[i, j, :]
    return combinedImage

#reading the image samples-----------------------------------------
Observation1=pd.read_csv("Trans1.csv", header=None).to_numpy()
Observation2=pd.read_csv("Trans2.csv", header=None).to_numpy()

#calculate the transform matrixes----------------------------------
T12=ComputeTransformationParameter(Observation1)
T32=ComputeTransformationParameter(Observation2)

#reading the images-------------------------------------------------
img1=cv2.imread("1190.jpg")
img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2=cv2.imread("1191.jpg")
img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img3=cv2.imread("1192.jpg")
img3=cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

#crate panorama (wuthout pyramids)----------------------------------
PanR,R1,R2,R3,_,_,_=panoramaMaking(img1[:, :, 0],img2[:, :, 0],img3[:, :, 0],T12,T32)
PanG,G1,G2,G3,_,_,_=panoramaMaking(img1[:, :, 1],img2[:, :, 1],img3[:, :, 1],T12,T32)
PanB,B1,B2,B3,BinaryImage1, BinaryImage2, BinaryImage3=panoramaMaking(img1[:, :, 2],img2[:, :, 2],img3[:, :, 2],T12,T32)

NewImage=cv2.merge((PanR/255,PanG/255,PanB/255))
Img1Trans=cv2.merge((R1/255,G1/255,B1/255))
Img2Trans=cv2.merge((R2/255,G2/255,B2/255))
Img3Trans=cv2.merge((R3/255,G3/255,B3/255))

plt.imshow(NewImage)
plt.show()

#careate Peramid for each image -------------------------------------
Img1GausianP=peramidGL5(Img1Trans,False)
Img2GausianP=peramidGL5(Img2Trans,False)
Img3GausianP=peramidGL5(Img3Trans,False)

#Img1GausianP.plot()
Img2GausianP.plot()
#Img3GausianP.plot()
plt.show()

#Create the pyramid for the binary image ----------------------------
BinP1=BinGausinaPeramid(BinaryImage1,False)
BinP2=BinGausinaPeramid(BinaryImage2,False)
BinP3=BinGausinaPeramid(BinaryImage3,False)


#Create the pyramid of the panorama by combined the images in all the levels.
Level5=CombineImages(Img1GausianP.level5,Img2GausianP.level5,Img3GausianP.level5,BinP1.level5,BinP2.level5,BinP3.level5)
Level5 = (Level5 - Level5.min()) / ((Level5 - Level5.min()).max())  #normalized the level between 0 and 1
Level54 = CombineImages(Img1GausianP.level5L,Img2GausianP.level5L,Img3GausianP.level5L,BinP1.level4,BinP2.level4,BinP3.level4)
Level4=CombineImages(Img1GausianP.level4L,Img2GausianP.level4L,Img3GausianP.level4L,BinP1.level3,BinP2.level3,BinP3.level3)
Level4 = (Level4 - Level4.min()) / ((Level4 - Level4.min()).max())
Level3=CombineImages(Img1GausianP.level3L,Img2GausianP.level3L,Img3GausianP.level3L,BinP1.level2,BinP2.level2,BinP3.level2)
Level3 = (Level3 - Level3.min()) / ((Level3 - Level3.min()).max())
Level2=CombineImages(Img1GausianP.level2L,Img2GausianP.level2L,Img3GausianP.level2L,BinP1.level1,BinP2.level1,BinP3.level1)
Level2 = (Level2 - Level2.min()) / ((Level2 - Level2.min()).max())
Level1=CombineImages(Img1GausianP.level1L,Img2GausianP.level1L,Img3GausianP.level1L,BinP1.level0,BinP2.level0,BinP3.level0)
Level1 = (Level1 - Level1.min()) / ((Level1 - Level1.min()).max())


#expend the combined pyramid and get the panorama ----------------------
p1=Level54+expend(Level5,Level54.shape)
p0=Level4+expend(p1,Level4.shape)
p2=Level3+expend(p0,Level3.shape)
p3 =Level2+expend(p2,Level2.shape)
PANORAMA_P=Level1+expend(p3,Level1.shape)
PANORAMA_P = (PANORAMA_P - PANORAMA_P.min()) / ((PANORAMA_P - PANORAMA_P.min()).max())

plt.imshow(PANORAMA_P)
plt.show()

#Addin a image on the wall-------------------------------------------------
Mona=cv2.imread("MonaLisaResize.jpg")
Mona=cv2.cvtColor(Mona, cv2.COLOR_BGR2RGB)
Mona = (Mona - Mona.min()) / ((Mona - Mona.min()).max())

Observation3=pd.read_csv("4Mona.csv", header=None).to_numpy()
TMona=ComputeTransformationParameter(Observation3)

Mona_Trans=np.zeros(PANORAMA_P.shape)
Mona_Bin=np.zeros(PANORAMA_P.shape[:2])

for i in range(NewImage.shape[0]):
    for j in range(NewImage.shape[1]):
        p=np.array([[j],[i],[1]])
        pm=np.linalg.inv(TMona)@p
        u = pm[0, 0] / pm[2, 0]
        v = pm[1, 0] / pm[2, 0]
        if 0 < u < len(Mona[0]) and 0 < v < len(Mona):
            NewImage[i, j, :] = Mona[int(v), int(u), :]
            Mona_Trans[i, j, :] = Mona[int(v), int(u), :]
            Mona_Bin[i, j] = 1


plt.imshow(NewImage)
plt.show()

#creating a mask and transformed mona liza
MonaP=peramidGL5(Mona_Trans,False)
MonaBinP=BinGausinaPeramid(Mona_Bin,True)


#Adding the mona liza in all the levels of the pyramid
for i in range(Level5.shape[0]):
    for j in range(Level5.shape[1]):
        if MonaBinP.level5[i, j] != 0:
            Level5[i, j, :] = MonaP.level5[i, j, :]

for i in range(Level54.shape[0]):
    for j in range(Level54.shape[1]):
        if MonaBinP.level4[i, j] != 0:
            Level54[i, j, :] = MonaP.level5L[i, j, :]


for i in range(Level4.shape[0]):
    for j in range(Level4.shape[1]):
        if MonaBinP.level3[i, j] != 0:
            Level4[i, j, :] = MonaP.level4L[i, j, :]

for i in range(Level3.shape[0]):
    for j in range(Level3.shape[1]):
        if MonaBinP.level2[i, j] != 0:
            Level3[i, j, :] = MonaP.level3L[i, j, :]


for i in range(Level2.shape[0]):
    for j in range(Level2.shape[1]):
        if MonaBinP.level1[i, j] != 0:
            Level2[i, j, :] = MonaP.level2L[i, j, :]

for i in range(Level1.shape[0]):
    for j in range(Level1.shape[1]):
        if MonaBinP.level0[i, j] != 0:
            Level1[i, j, :] = MonaP.level1L[i, j, :]


#expend the combined pyramid and get the panorama ----------------------
p1=Level54+expend(Level5,Level54.shape)
p0=Level4+expend(p1,Level4.shape)
p2=Level3+expend(p0,Level3.shape)
p3 =Level2+expend(p2,Level2.shape)
PANORAMA_P=Level1+expend(p3,Level1.shape)
PANORAMA_P = (PANORAMA_P - PANORAMA_P.min()) / ((PANORAMA_P - PANORAMA_P.min()).max())

plt.imshow(PANORAMA_P)
plt.show()
