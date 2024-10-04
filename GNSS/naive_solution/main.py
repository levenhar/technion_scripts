import matplotlib.pyplot as plt
import Reader
import numpy as np
import MatrixMethods as MM
import regration as lin
from sklearn.neighbors import LocalOutlierFactor


def Ecal(Mj,e):
    Eold=Mj
    Enew=Mj+e*np.sin(Mj)
    c=0
    while np.abs(Enew-Eold)>10**-12:
        Eold=Enew
        Enew=Mj+e*np.sin(Enew)
        c+=1
    return Enew

def SlocatinCal(NN,t):
    we=7.292115147*10**-5
    a=NN[2,3]**2
    e=NN[2,1]
    u=3.986005054321*10**14
    tj= t - NN[3, 0]
    Mj= NN[1, 3] + (np.sqrt(u / (a ** 3)) + NN[1, 2]) * tj
    Ej=Ecal(Mj,e)
    fj=np.arctan2((np.sqrt(1-e**2)*np.sin(Ej)),(np.cos(Ej)-e))
    if fj<0:
        fj=fj+2*np.pi
    Omegaj= NN[3, 2] + (NN[4, 3] - we) * tj - we * NN[3, 0]
    q=2 * (NN[4, 2] + fj)
    wj= NN[4, 2] + fj + NN[2, 0] * np.cos(q) + NN[2, 2] * np.sin(q)
    rj= a * (1-e*np.cos(Ej)) + NN[4, 1] * np.cos(q) + NN[1, 1] * np.sin(q)
    ij= NN[4, 0] + NN[5, 0] * tj + NN[3, 1] * np.cos(q) + NN[3, 3] * np.sin(q)

    xj=rj*np.cos(wj)
    yj=rj*np.sin(wj)

    Xj=xj*np.cos(Omegaj)-yj*np.sin(Omegaj)*np.cos(ij)
    Yj=xj*np.sin(Omegaj)+yj*np.cos(Omegaj)*np.cos(ij)
    Zj=yj*np.sin(ij)

    dti = NN[0, 1] + NN[0, 2]*tj + NN[0,3]*tj**2

    return np.array([Xj,Yj,Zj]).T, dti


def XYZcalculate(Observations,satelliteNum,aprx,NfileSulotion,NfileOrder,dt):
    C1=Observations[:,3]
    dt=np.array(dt)
    if len(dt)<12:
        mm=np.zeros(12-len(dt))
        dt=np.concatenate((dt,mm))
    #dt = Observations[:, 3]
    satlocation=np.array([0,0,0]).reshape((1,3))
    removeList=[]
    dtGoodOrder=[]
    for ii in range(len(satelliteNum)):
        for jj in range(len(NfileOrder)):
            if satelliteNum[ii] == NfileOrder[jj] and NfileOrder[jj]!="04" and C1[ii]!=0:
                satlocation=np.concatenate((satlocation,NfileSulotion[jj].reshape(1,3)),axis=0)
                dtGoodOrder.append(dt[jj])
                break
        else:
            removeList.append(ii)
    satlocation=np.delete(satlocation,0,0)
    C1=np.delete(C1,removeList)
    #dt = np.delete(dt, removeList)
    old = np.array([np.inf, np.inf, np.inf]).reshape(3, 1)

    c = 3 * 10 ** 8
    aprx = aprx.astype(np.float)
    cc=0
    while np.linalg.norm(aprx - old)>10 and cc<100:
        L = np.zeros((len(satlocation), 1))
        A = np.zeros((len(satlocation), 4))
        old = aprx
        for j in range(len(C1)):
            satL=satlocation[j].reshape((3,1))
            satL.astype(np.float)
            satL=satL*10**3
            ru0=np.linalg.norm(aprx-satL)
            ll=C1[j]-ru0+c*dtGoodOrder[j]
            L[j]=ll
            A[j,:]=np.array([-(satL[0,0]-aprx[0,0])/ru0,-(satL[1,0]-aprx[1,0])/ru0,-(satL[2,0]-aprx[2,0])/ru0,-1])
        try:
            X=np.linalg.inv(A.T@A)@(A.T@L)
            cc+=1
        except:
            pass
        if cc == 100:
            print("100")
            continue
        aprx=aprx+X[:3,:].reshape(3,1)
    V = A @ X - L
    s = (V.T@V/(len(satlocation)-4))[0,0]
    return aprx , s

N, Dates = Reader.readNfile("elat042a18n.txt")

NfileSulotion=np.array([0,0,0]).reshape((1,3))
SatNumber=[]
dt=[]
for i in range(len(Dates)):
    info=Dates[i,0].split(" ")
    for j in info:
        if j=="":
            info.remove("")

    if info[1:]==["18","2","11","2","0","0.0"]:
        SatNumber.append(info[0])
        XYZ , dti = SlocatinCal(N[:, :, i], N[3, 0, i])
        NfileSulotion=np.concatenate((NfileSulotion,(XYZ*10**-3).reshape((1,3))) , axis=0)
        dt.append(dti)


CoordinatesAll, Information, Satell=Reader.readsp3("igs19880sp3.txt")

TrueCoordinate=np.array([0,0,0,0]).reshape((1,4))
for i in range(len(Information)):
    if Information[i]==["*","2018","2","11","2","0","0.00000000"]:
        for sat in SatNumber:
            if len(sat)==1:
                sat="0"+sat
            for j in range(len(Satell)):
                if Satell[j][2:]==sat:
                    break
            TrueCoordinate=np.concatenate((TrueCoordinate,CoordinatesAll[j,:,i].reshape((1,4))),axis=0)
        break

TrueCoordinate=np.delete(TrueCoordinate,0,0)
NfileSulotion=np.delete(NfileSulotion,0,0)

fig=plt.figure()
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)

ax1.scatter(TrueCoordinate[:,0],NfileSulotion[:,0],color="b")
ax2.scatter(TrueCoordinate[:,1],NfileSulotion[:,1],color="r")
ax3.scatter(TrueCoordinate[:,2],NfileSulotion[:,2],color="g")
#plt.legend(("X","Y","Z"))

ax1.set_title("X Coordinates")
ax1.set_xlabel("True Coordinate from sp3 [km]")
ax1.set_ylabel("Calculate Coordinate from n file [km]")
ax2.set_title("Y Coordinates")
ax2.set_xlabel("True Coordinate from sp3 [km]")
ax2.set_ylabel("Calculate Coordinate from n file [km]")
ax3.set_title("Z Coordinates")
ax3.set_xlabel("True Coordinate from sp3 [km]")
ax3.set_ylabel("Calculate Coordinate from n file [km]")

bx=lin.estimate_coef(TrueCoordinate[:,0],NfileSulotion[:,0])
by=lin.estimate_coef(TrueCoordinate[:,1],NfileSulotion[:,1])
bz=lin.estimate_coef(TrueCoordinate[:,2],NfileSulotion[:,2])

lin.plot_regression_line(TrueCoordinate[:,0],bx,"b",ax1)
lin.plot_regression_line(TrueCoordinate[:,1],by,"r",ax2)
lin.plot_regression_line(TrueCoordinate[:,2],bz,"g",ax3)

plt.show()

Diff=TrueCoordinate[:,:3]-NfileSulotion

print(SatNumber)
MM.PrintMatrix(NfileSulotion)
MM.PrintMatrix(Diff)

print(bx)
print(by)
print(bz)

Observations, satelliteNum, info , AproxPosition = Reader.readOfile("elat042a18o.txt")
trueVal=np.array([4555028.330,3180067.508,3123164.667]).reshape((3,1))

for i in range(1,len(info)):
    if info[i,3]=="2" and info[i,4]=="0" and info[i,5]=="0.0000000":
        LOCATION , S=XYZcalculate(Observations[:,:,i],satelliteNum[i],AproxPosition[:3].reshape((3,1)),NfileSulotion,SatNumber,dt)
        break
print("Position by n-file")
MM.PrintMatrix(LOCATION)
print("Error position by n-file")
MM.PrintMatrix((LOCATION-trueVal))

c=0
for s in range(len(Satell)):
    Satell[s]=Satell[s].replace("PG","")


station=np.zeros((96,3))
sigmas=np.zeros((96,1))
for i in range(1,Observations.shape[2],180):
    dt=np.zeros((len(Observations[:,0,1]),1))
    sateLocation=np.zeros((len(Observations[:,0,1]),3))
    Coor, s=XYZcalculate(Observations[:, :, i], satelliteNum[i], AproxPosition[:3].reshape((3, 1)), CoordinatesAll[:, :3, c], Satell, CoordinatesAll[:, 3, c]*10**-6)
    station[c,:]=Coor.T
    sigmas[c, 0] = s**0.5
    c+=1

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([0], [0], [0],color="black",s=35)
ax.scatter(station[:,0]-trueVal[0,0],station[:,1]-trueVal[1,0],station[:,2]-trueVal[2,0],color="g",s=20)

plt.show()


lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
outliers = lof.fit_predict(station)
deletRows=np.where(outliers == -1)[0]

station=np.delete(station,deletRows,axis=0)

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([0], [0], [0],color="black",s=35)
ax.scatter(station[:,0]-trueVal[0,0],station[:,1]-trueVal[1,0],station[:,2]-trueVal[2,0],color="g",s=20)

plt.show()
'''station[:,0]=station[:,0]-trueVal[0,0]
station[:,1]=station[:,1]-trueVal[1,0]
station[:,2]=station[:,2]-trueVal[2,0]'''

#MM.PrintMatrix(station)
print("Average Position")
print(np.average(station[:,0]))
print(np.average(station[:,1]))
print(np.average(station[:,2]))
print()
print("Average Error")
print(np.average(station[:,0])-trueVal[0,0])
print(np.average(station[:,1])-trueVal[1,0])
print(np.average(station[:,2])-trueVal[2,0])
print()
print("Average absolute Error")
print(np.average(abs(station[:,0]-trueVal[0,0])))
print(np.average(abs(station[:,1]-trueVal[1,0])))
print(np.average(abs(station[:,2]-trueVal[2,0])))
print()
print("STD")
print(np.std(station[:,0]))
print(np.std(station[:,1]))
print(np.std(station[:,2]))
print()
print("Average sigma aposteriori")
print(np.average(sigmas))