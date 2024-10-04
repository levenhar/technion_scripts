import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal

#Q1
#------------reading the Image--------
image=cv2.imread("NYC.jpeg",0)
plt.imshow(image,cmap="gray")
plt.show()
#--------------------------------------

#-------Calculate Fourier transform------------
G=np.fft.fft2(image)
G=np.fft.fftshift(G)

plt.imshow(np.log(1+np.abs(G)),cmap="pink")
plt.show()
#--------------------------------------------

#------------Low Pass Filter-----------------
Boxfilter=np.ones((len(image),len(image[0])))
Boxfilter[len(image)//2-100:len(image)//2+100,len(image[0])//2-100:len(image[0])//2+100]=np.zeros((200,200))
Boxfilter=1-Boxfilter

NewCofi=np.multiply(Boxfilter,G)
plt.imshow(np.log(1+np.abs(NewCofi)),cmap="pink")
plt.show()

plt.figure()
lopassImage=np.abs((np.fft.ifft2(NewCofi)))
plt.imshow(lopassImage, cmap="gray")
plt.show()
print(f"ImageSize - {lopassImage.shape}")
#-----------------------------------------------

#---------cut the Imagein frequency space------------------------

NewCofi_cut=NewCofi[len(image)//2-100:len(image)//2+100,len(image[0])//2-100:len(image[0])//2+100]
plt.imshow(np.log(1+np.abs(NewCofi_cut)),cmap="pink")
plt.title("Cutting the frequency space")
plt.show()

plt.figure()
lopassImage_cut=np.abs((np.fft.ifft2(NewCofi_cut)))
plt.imshow(lopassImage_cut, cmap="gray")
plt.title("Image after cutting")
plt.show()
print(f"ImageSize - {lopassImage_cut.shape}")


#----------High Pass Filter Circle---------------------
highpass=np.ones(Boxfilter.shape)
ci,cj=highpass.shape[1]/2,highpass.shape[0]/2
cr=100
# Create index arrays to the filter
I,J=np.meshgrid(np.arange(highpass.shape[0]),np.arange(highpass.shape[1]))
# calculate distance of all points to centre
dist=np.sqrt((I-ci)**2+(J-cj)**2)
# Assign value of 1 to those points where dist<cr:
highpass[np.where(dist<cr)]=0

NewCofi1=np.multiply(highpass,G)
plt.imshow(np.log(1+np.abs(NewCofi1)),cmap="pink")
plt.title("Ideal High pass Filter ")
plt.show()

HighpassImage=np.abs(np.fft.ifft2(NewCofi1))

plt.imshow(HighpassImage,cmap="gray")
plt.show()
print(f"ImageSize - {HighpassImage.shape}")
#------------------------------------------------


#----------Low Pass Filter Circle---------------------
lowpass=1-highpass

NewCofi2=np.multiply(lowpass,G)
plt.imshow(np.log(1+np.abs(NewCofi2)),cmap="pink")
plt.title("Ideal High pass Filter ")
plt.show()

LowpassImage=np.abs(np.fft.ifft2(NewCofi2))

plt.imshow(LowpassImage,cmap="gray")
plt.show()
print(f"ImageSize - {LowpassImage.shape}")
#------------------------------------------------

#-----------Convolution Gausian------------------
NewImage = cv2.GaussianBlur(image,(5,5),1)
plt.imshow(NewImage,cmap="gray")
plt.title("Image after Gaussian filter")
plt.show()

#-----------Gausian by Fourier-------------------
GausianBox = np.outer(signal.gaussian(5, 1), signal.gaussian(5, 1))  # Gaussian filter with size 5x5 and std=1
GausianBox = (1/np.sum(GausianBox))*GausianBox

G=np.fft.fft2(image)
GausianBoxF = np.fft.fft2(GausianBox, (image.shape[0], image.shape[1]))

GausianCofi=G*GausianBoxF


BlurImage=np.abs((np.fft.ifft2(GausianCofi)))
plt.imshow(BlurImage,cmap="gray")
plt.title("Image after Gaussian Fourier")
plt.show()