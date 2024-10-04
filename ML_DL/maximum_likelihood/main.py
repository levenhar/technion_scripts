import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.patches as patches
import pandas as pd
import colorsys

class group():
    def __init__(self,sampels,PixNum,color=None):
        self.sampels = sampels
        self.Mean=np.array([np.mean(sampels[:,0]),np.mean(sampels[:,1]), np.mean(sampels[:,2])]).reshape((3,1))
        self.C=np.cov(sampels.T)
        self.Pix=PixNum
        if color is None:
            self.color=self.Mean
        else:
            self.color=color

    def disFromMean2(self,point):
        r=(point[0,0]-self.Mean[0,0])**2+(point[1,0]-self.Mean[1,0])**2+(point[2,0]-self.Mean[2,0])**2
        return r

    def maximumLikeX(self,point):
        v=point-self.Mean

        g=np.log(len(point)/self.Pix)-0.5*np.log(np.linalg.det(self.C))-0.5*v.T@np.linalg.inv(self.C)@v
        return g[0,0]

    def samplePlot(self,ax,color1):
        ax.scatter(self.sampels[0:-1:100, 0], self.sampels[0:-1:100, 1], self.sampels[0:-1:100, 2], color=color1)

    def __repr__(self):
        return f"the mean RGB value is {self.Mean}, and the accuracy is {np.diag(self.C)} \n \n"

myImage=cv2.imread("RGBimage.jpeg")
myImage=cv2.cvtColor(myImage, cv2.COLOR_BGR2RGB)


Sky1=myImage[18:74, 647: 725]
Sky1=Sky1.reshape((Sky1.shape[0]*Sky1.shape[1],3))
Sky2=myImage[174:204, 455:491]
Sky2=Sky2.reshape((Sky2.shape[0]*Sky2.shape[1],3))
Sky3=myImage[54:139, 820:996]
Sky3=Sky3.reshape((Sky3.shape[0]*Sky3.shape[1],3))
Sky4=myImage[222:233, 540:691]
Sky4=Sky4.reshape((Sky4.shape[0]*Sky4.shape[1],3))
Sky_S=np.concatenate((Sky1,Sky2,Sky3,Sky4),axis=0)

Poll1=myImage[285:413, 717:751]
Poll1=Poll1.reshape((Poll1.shape[0]*Poll1.shape[1],3))
Poll2=myImage[494:550, 367:393]
Poll2=Poll2.reshape((Poll2.shape[0]*Poll2.shape[1],3))
Polls_S=np.concatenate((Poll1,Poll2),axis=0)

Sand1=myImage[501:573, 164:233]
Sand1=Sand1.reshape((Sand1.shape[0]*Sand1.shape[1],3))
Sand2=myImage[567:654, 523:656]
Sand2=Sand2.reshape((Sand2.shape[0]*Sand2.shape[1],3))
Sand_S=np.concatenate((Sand1,Sand2),axis=0)

Rocks1=myImage[400:417, 90:122]
Rocks1=Rocks1.reshape((Rocks1.shape[0]*Rocks1.shape[1],3))
Rocks2=myImage[387:398, 210:227]
Rocks2=Rocks2.reshape((Rocks2.shape[0]*Rocks2.shape[1],3))
Rocks_S=np.concatenate((Rocks1,Rocks2),axis=0)

Clouds1=myImage[72:103, 348:380]
Clouds1=Clouds1.reshape((Clouds1.shape[0]*Clouds1.shape[1],3))
Clouds2=myImage[267:280, 499:520]
Clouds2=Clouds2.reshape((Clouds2.shape[0]*Clouds2.shape[1],3))
Clouds3=myImage[935:970, 6:30]
Clouds3=Clouds3.reshape((Clouds3.shape[0]*Clouds3.shape[1],3))
Clouds_S=np.concatenate((Clouds1,Clouds2,Clouds3),axis=0)

OrengeSky=myImage[338:358, 126:283]
OrengeSky_S=OrengeSky.reshape((OrengeSky.shape[0]*OrengeSky.shape[1],3))

Sun=myImage[301:310, 149:156]
Sun_S=Sun.reshape((Sun.shape[0]*Sun.shape[1],3))

Sidewalk=myImage[635:753, 14:105]
Sidewalk_S=Sidewalk.reshape((Sidewalk.shape[0]*Sidewalk.shape[1],3))

Sea1=myImage[369:441, 484:666]
Sea1=Sea1.reshape((Sea1.shape[0]*Sea1.shape[1],3))
Sea2=myImage[397:446, 797:1007]
Sea2=Sea2.reshape((Sea2.shape[0]*Sea2.shape[1],3))
Sea_S=np.concatenate((Sea1,Sea2),axis=0)


#------------------------------------------------------------------------

plt.imshow(myImage)

rect=patches.Rectangle((647,74),725-647,-(74-18), color="black", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((445,204),491-445,-(204-174), color="black", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((717,413),751-717,-(413-285), color="b", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((367,550),393-367,-(550-494), color="b", fill=False)
plt.gca().add_patch(rect)

rect=patches.Rectangle((164,573),233-164,-(573-501), color="r", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((523,654),656-523,-(654-567), color="r", fill=False)
plt.gca().add_patch(rect)

rect=patches.Rectangle((90,417),122-90,-(417-400), color="g", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((210,398),227-210,-(398-387), color="g", fill=False)
plt.gca().add_patch(rect)


rect=patches.Rectangle((348,103),380-348,-(103-72), color="m", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((499,280),520-499,-(280-267), color="m", fill=False)
plt.gca().add_patch(rect)

rect=patches.Rectangle((126,358),283-126,-(358-338), color="c", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((149,310),156-149,-(310-301), color="gold", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((14,753),105-14,-(753-635), color="peru", fill=False)
plt.gca().add_patch(rect)
rect=patches.Rectangle((484,441),666-484,-(441-369), color="crimson", fill=False)
plt.gca().add_patch(rect)

plt.show()
#-------------------------------------------------------------------------
Sky=group(Sky_S, myImage.shape[0]*myImage.shape[1])
Polls=group(Polls_S, myImage.shape[0]*myImage.shape[1])
Sand=group(Sand_S, myImage.shape[0]*myImage.shape[1])
Rocks=group(Rocks_S, myImage.shape[0]*myImage.shape[1])
Clouds=group(Clouds_S, myImage.shape[0]*myImage.shape[1])
OrengeSky=group(OrengeSky_S, myImage.shape[0]*myImage.shape[1])
Sun=group(Sun_S, myImage.shape[0]*myImage.shape[1])
Sidewalk=group(Sidewalk_S, myImage.shape[0]*myImage.shape[1])
Sea=group(Sea_S, myImage.shape[0]*myImage.shape[1])
#--------------------------------------------------------------------------
Groups=[Sky,Polls,Sand,Rocks,Clouds,OrengeSky,Sun,Sidewalk,Sea]
MinDisImage=np.zeros(myImage.shape,dtype=int)
MaxLikeImage=np.zeros(myImage.shape,dtype=int)
'''MaxLikeImageB=np.zeros(myImage.shape[:2])
MinDisImageB=np.zeros(myImage.shape[:2])'''
#----------------------------------------------------------------
colors=["black","b","r","g","m","c","gold","peru","crimson"]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for G in range(len(Groups)):
    Groups[G].samplePlot(ax,colors[G])
    print(Groups[G])
plt.show()
#-----------------------------------------------------------------
for i in range(myImage.shape[0]):
    for j in range(myImage.shape[1]):
        G = np.array([0,0]).reshape((1,2))
        D = np.array([0,0]).reshape((1,2))
        for k in range(len(Groups)):
            p=myImage[i,j,:].reshape((3,1))
            d=Groups[k].disFromMean2(p)
            g=Groups[k].maximumLikeX(p)
            G = np.concatenate((G, np.array([g, k]).reshape((1,2))),axis=0)
            D = np.concatenate((D, np.array([d, k]).reshape((1,2))),axis=0)
        G=G[1:]
        D=D[1:]
        maxx=-np.inf
        maxK=0
        minn=np.inf
        minK=0
        for m in range(len(G)):
            if G[m,0] > maxx:
                maxx=G[m,0]
                maxK=m
            if D[m,0] < minn:
                minn=D[m,0]
                minK = m
        MinDisImage[i,j,:]=np.array(Groups[minK].color.reshape(3), dtype=int)
        MaxLikeImage[i, j, :] = np.array(Groups[maxK].color.reshape(3), dtype=int)
'''MaxLikeImageB[i, j] = maxK / 8
MinDisImageB[i, j] = minK / 8'''

Legend=np.zeros((90,90,3),dtype=int)
Legend[0:30,0:30,0]=np.ones((30,30))*int(Sky.color[0,0])
Legend[0:30,0:30,1]=np.ones((30,30))*int(Sky.color[1,0])
Legend[0:30,0:30,2]=np.ones((30,30))*int(Sky.color[2,0])
cv2.putText(Legend,"Sky",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0))

Legend[30:60,0:30,0]=np.ones((30,30))*int(Polls.color[0,0])
Legend[30:60,0:30,1]=np.ones((30,30))*int(Polls.color[1,0])
Legend[30:60,0:30,2]=np.ones((30,30))*int(Polls.color[2,0])
cv2.putText(Legend,"Polls",(5,45),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255))

Legend[60:,0:30,0]=np.ones((30,30))*int(Sand.color[0,0])
Legend[60:,0:30,1]=np.ones((30,30))*int(Sand.color[1,0])
Legend[60:,0:30,2]=np.ones((30,30))*int(Sand.color[2,0])
cv2.putText(Legend,"Sand",(5,75),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0))

Legend[0:30,30:60,0]=np.ones((30,30))*int(Rocks.color[0,0])
Legend[0:30,30:60,1]=np.ones((30,30))*int(Rocks.color[1,0])
Legend[0:30,30:60,2]=np.ones((30,30))*int(Rocks.color[2,0])
cv2.putText(Legend,"Rocks",(35,15),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0))

Legend[30:60,30:60,0]=np.ones((30,30))*int(Clouds.color[0,0])
Legend[30:60,30:60,1]=np.ones((30,30))*int(Clouds.color[1,0])
Legend[30:60,30:60,2]=np.ones((30,30))*int(Clouds.color[2,0])
cv2.putText(Legend,"Clouds",(35,45),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0))

Legend[60:,30:60,0]=np.ones((30,30))*int(OrengeSky.color[0,0])
Legend[60:,30:60,1]=np.ones((30,30))*int(OrengeSky.color[1,0])
Legend[60:,30:60,2]=np.ones((30,30))*int(OrengeSky.color[2,0])
cv2.putText(Legend,"OrengeSky",(35,75),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0))

Legend[0:30,60:,0]=np.ones((30,30))*int(Sun.color[0,0])
Legend[0:30,60:,1]=np.ones((30,30))*int(Sun.color[1,0])
Legend[0:30,60:,2]=np.ones((30,30))*int(Sun.color[2,0])
cv2.putText(Legend,"Sun",(65,15),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0))

Legend[30:60,60:,0]=np.ones((30,30))*int(Sidewalk.color[0,0])
Legend[30:60,60:,1]=np.ones((30,30))*int(Sidewalk.color[1,0])
Legend[30:60,60:,2]=np.ones((30,30))*int(Sidewalk.color[2,0])
cv2.putText(Legend,"Sidewalk",(65,45),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0))

Legend[60:,60:,0]=np.ones((30,30))*int(Sea.color[0,0])
Legend[60:,60:,1]=np.ones((30,30))*int(Sea.color[1,0])
Legend[60:,60:,2]=np.ones((30,30))*int(Sea.color[2,0])
cv2.putText(Legend,"Sea",(65,75),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0))

fig, ([[ax1,ax2],[ax3,ax4]]) = plt.subplots(2,2)

ax1.imshow(MinDisImage)
ax2.imshow(MaxLikeImage)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_title("Minimum Distance Calssification")
ax2.set_title("Maximum Likelihood Calssification")

ax3.set_title("RGB Image")
ax3.imshow(myImage)
ax3.set_xticks([])
ax3.set_yticks([])

ax4.set_title("Legend")
ax4.imshow(Legend)
ax4.set_xticks([])
ax4.set_yticks([])

plt.show()
