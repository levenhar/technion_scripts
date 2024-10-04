import matplotlib.pyplot as plt
import Reader
import numpy as np
import MatrixMethods as MM
import regration as lin
from sklearn.neighbors import LocalOutlierFactor
import datetime
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import pandas as pd
import re
from scipy.interpolate import interp1d
from numpy import cos,sin,tan,arctan2,arctan
from scipy.io import netcdf


def find_closest_index(lst, value):
    """
    Returns the index of the element in lst that is closest to the given value.
    """
    return min(range(len(lst)), key=lambda m: abs(lst[m] - value))

def IONEXfinder(ionexfile,point):
    # find the value in ionexfile according to a point at 2 am
    ionexData=ionexfile[:,:,2]
    i=-int((point[0]-87.5)/2.5)
    j=int((point[1]+180)/5)
    return ionexData[i,j]


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

def XYZcalculate(Observ,aprx=None,satlocation=None,dt=None,Tropo=np.array([]),Iono=np.array([]),IonoFree=False,GeomertyFree=False):
    if GeomertyFree:
        C1 = Observ[:8, 0]
        P2 = Observ[:8, 3]
        f1 = 1575.42
        f2 = 1227.6
        P4=C1-P2
        return P4*f2**2/(f2**2-f1**2)
    if IonoFree:
        C1=Observ[:8,0]
        P2=Observ[:8,3]
        f1=1575.42
        f2=1227.6
        C1=f1**2/(f1**2-f2**2)*C1-f2**2/(f1**2-f2**2)*P2
    else:
        C1=Observ[:8,0]
    dt=np.array(dt)

    old = np.array([np.inf, np.inf, np.inf]).reshape(3, 1)

    c = 3 * 10 ** 8
    aprx = aprx.astype(np.float)
    cc=0
    while np.linalg.norm(aprx - old)>0.1 and cc<100:
        L = np.zeros((len(satlocation), 1))
        A = np.zeros((len(satlocation), 4))
        old = aprx
        for j in range(len(C1)):
            satL=satlocation[j].reshape((3,1))
            satL.astype(np.float)
            satL=satL*10**3
            ru0=np.linalg.norm(aprx-satL)
            ll=C1[j]-ru0+c*dt[j]
            L[j]=ll
            A[j,:]=np.array([-(satL[0,0]-aprx[0,0])/ru0,-(satL[1,0]-aprx[1,0])/ru0,-(satL[2,0]-aprx[2,0])/ru0,-1])
        if Tropo.any():
            L=L+Tropo
        if Iono.any():
            L=L-Iono
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

def XYZ2LLh(Point):
    a=6378137
    f=1/298.25722356
    e=np.sqrt(2*f-f**2)
    L=np.arctan2(Point[1],Point[0])
    phi=np.arctan(Point[2]/(np.sqrt(Point[0]**2+Point[1]**2))*(1-e**2)**-1)
    phiold=np.inf
    while np.linalg.norm(phi-phiold)>10**-8:
        phiold=phi
        N=a/np.sqrt(1-e**2*np.sin(phi)**2)
        h=np.sqrt(Point[0]**2+Point[1]**2)/np.cos(phi)-N
        phi=np.arctan(Point[2]/(np.sqrt(Point[0]**2+Point[1]**2))*(1-e**2*N/(N+h))**-1)
    return np.array([np.rad2deg(phi),np.rad2deg(L),h])

def Pmatrix(f,l):
    P=np.array([[cos(f)*cos(l), cos(f)*sin(l), sin(f)], [-sin(l), cos(l), 0], [-sin(f)*cos(l), -sin(f)*sin(l), cos(f)]])
    return P

#calculate lat long and h   ################################################################################
Point=np.array([4555028.330, 3180067.508, 3123164.667])
LLH=XYZ2LLh(Point)

#sp3 reading #################################################################################################
sp3_data = pd.read_csv("igs19880sp3.csv", skiprows=20, header=None, delim_whitespace=True,usecols=list(range(9)))
sp3_dataSatNames = pd.read_csv("igs19880sp3.csv", header=None, delim_whitespace=True,nrows=4).to_numpy()
SatNames=sp3_dataSatNames[2,2] + sp3_dataSatNames[3,1]
SatNames=SatNames.split("G")
for i in range(len(SatNames)):
    SatNames[i]="PG"+SatNames[i]
del SatNames[0]
del SatNames[3]
sp3_data = sp3_data.drop(0)
sp3_data = sp3_data.drop(1)
sp3_data = sp3_data.to_numpy()
sp3_data = sp3_data[sp3_data[:,0]!="PG04",:]

Timearray=[datetime.datetime(2018, 2, 11, 0, 0, 0)]
for i in range(1,96):
    Timearray.append((Timearray[i-1]+datetime.timedelta(minutes=15)))

for i in range(0,96):
    Timearray[i]=np.float(Timearray[i].timestamp())
Timearray=np.array(Timearray).T
SetallitePosition=[]
T_error=[]

#O-file reading ########################################################################
Observations, satelliteNum, info , AproxPosition = Reader.readOfile("elat042a18o.txt")
trueVal=np.array([4555028.330,3180067.508,3123164.667]).reshape((3,1))

T=int(2*3600/5)+1
#calculate in spesic time ##################################################################
c=3*10**8
we=(15*np.pi/180)/3600  #rad/s
SP3Coordinate=np.zeros((8,3))
cc=0
for i, prn in enumerate(satelliteNum[T]):
    if prn =="04" or prn == "0.0":
        continue
    prn = "PG" + prn
    prn_data = sp3_data[sp3_data[:,0] == prn]
    tt=Observations[i,0,T]/c
    R=np.array([[np.cos(we*tt), np.sin(we*tt), 0], [-np.sin(we*tt), np.cos(we*tt), 0], [0, 0, 1]])
    data_time = datetime.datetime(2018, 2, 11, 2, 0, 0) - datetime.timedelta(seconds=tt)
    data_time = np.float(data_time.timestamp())

    j = find_closest_index(Timearray, data_time)

    SP3Coordinate[cc,:]=prn_data[j,1:4].astype(np.float)

    DATA4Poly = prn_data[j-6:j+7,1:5]
    DATA4Poly = DATA4Poly.astype(np.float)
    Times4Poly = Timearray[j-6:j+7]

    Times4Poly=Times4Poly.astype(np.float)

    #coeffs = np.polynomial.polynomial.polyfit(Times4Poly,DATA4Poly,4)
    coeffs = interp1d(Times4Poly, DATA4Poly.T, kind=3)
    pos_at_t = coeffs(data_time).T
    #pos_at_t = np.polynomial.polynomial.polyval(data_time, coeffs)
    #pos_at_t = prn_data[j,1:5].astype(np.float)

    t_error=pos_at_t[3]

    pos_at_t=R@(pos_at_t[:3].reshape((3,1)))
    SetallitePosition.append(pos_at_t)
    T_error.append(t_error)
    cc+=1

SetallitePosition=np.array(SetallitePosition)
SetallitePosition=SetallitePosition.reshape(SetallitePosition.shape[:2])
T_error = np.array(T_error)
Order=satelliteNum[T][0:8]

diff=SP3Coordinate-SetallitePosition

print("תוצאות של סעיף א - קורדינאטות לאחר תיקון זמן וסיבוב")
MM.PrintMatrix(SetallitePosition)
print("הפרש מקורדינאטות נתונות בsp3")
MM.PrintMatrix(diff)

Coor, s = XYZcalculate(Observations[:, :, T], AproxPosition[:3].reshape((3, 1)),SetallitePosition, T_error * 10 ** -6)

toPrint=np.concatenate((Coor,Coor-trueVal),axis=1)
MM.PrintMatrix(toPrint)

#Calculate elevation angle #################################################################

TT=Pmatrix(np.deg2rad(LLH[0]),np.deg2rad(LLH[1]))
Aprxx=AproxPosition[:3].reshape((3, 1)).astype(np.float)
uen=np.array(list(map(lambda v: TT@(v.reshape((3,1))-Aprxx) , SetallitePosition*10**3)))
uen=uen.reshape((8,3))
elev=np.array(list(map(lambda V: np.rad2deg(arctan(V[0]/np.sqrt(V[1]**2+V[2]**2))),uen))).reshape((8,1))
MM.PrintMatrix(elev)

# Calculate tropospheric delay ###############################################################
Ps=1013.25
Ts=293
es=11.7
h=13.061
phi=np.deg2rad(LLH[0])
f=1-0.00266*cos(2*phi)-0.00028*h
ZTDs=0.002277/f*(Ps+(1255/Ts+0.05)*es)

hw=11000
hd=40136+148.72*(Ts-273.15)
ZTDh=1.552*10**-5*Ps/Ts*(hd-h)+0.07465*es/(Ts**2)*(hw-h)
Re=6371
Hatm=15
r=Re/(Re+Hatm)
mf=list(map(lambda elv:(Re/Hatm+1)*(cos(np.arcsin(r*cos(np.deg2rad(elv))))-r*sin(np.deg2rad(elv))),elev))
TDh=np.array(list(map(lambda m:ZTDh*m,mf)))
TDs=np.array(list(map(lambda m:ZTDs*m,mf)))

MM.PrintMatrix(np.concatenate((TDh,TDs,abs(TDs-TDh)),axis=1))

CoorTropH, sTropH = XYZcalculate(Observations[:, :, T], AproxPosition[:3].reshape((3, 1)),SetallitePosition, T_error * 10 ** -6,TDh)

print("קרודינאטות לאחר התחשבות בעיכוב טרופוספרה H")
MM.PrintMatrix(np.concatenate((CoorTropH,CoorTropH-trueVal),axis=1))

CoorTropS, sTropS = XYZcalculate(Observations[:, :, T], AproxPosition[:3].reshape((3, 1)),SetallitePosition, T_error * 10 ** -6,TDs)

print("קרודינאטות לאחר התחשבות בעיכוב טרופוספרה S")
MM.PrintMatrix(np.concatenate((CoorTropS,CoorTropS-trueVal),axis=1))


###### Ionosphere delay #################################################################

H=350 #hight of  Ionosphere in km

Longtitude=LLH[1]*np.pi/180
Znit=90-elev
Ztag=np.array(list(map(lambda z: np.arcsin(Re/(Re+H)*sin(np.deg2rad(z))),Znit)))

print("Z and Ztag")
MM.PrintMatrix(np.concatenate((Znit,np.rad2deg(Ztag),Znit-np.rad2deg(Ztag)),axis=1))


Dz=np.deg2rad(Znit)-Ztag
Az=np.array(list(map(lambda V:arctan2(V[1],V[2]),uen))).reshape((8,1))
phiIPP=np.array(list(map(lambda dz,az: np.arcsin(sin(phi)*cos(dz)+cos(phi)*sin(dz)*cos(az)),Dz,Az)))
LongIPP=np.array(list(map(lambda f,dz,az:Longtitude+np.arcsin((sin(dz)*sin(az))/f),phiIPP,Dz,Az)))\

#calculte magnetic Position
phi0=78.3*np.pi/180
L0=291.0*np.pi/180
phiIPPm=np.array(list(map(lambda ff,ll:np.arcsin(sin(ff)*sin(phi0)+cos(ff)*cos(phi0)*cos(ll-L0)),phiIPP,LongIPP)))


print("Iono Position for each setalite - 350 KM")
MM.PrintMatrix(np.concatenate((np.rad2deg(phiIPP),np.rad2deg(phiIPPm),np.rad2deg(LongIPP)),axis=1))


df = pd.read_fwf("elat042a.18n.txt", header=None)
ar = df.to_numpy()
ar1=ar[2][0].replace("D","E",4).split(" ")
a=np.array([ar1[0],ar1[2],ar1[3],ar1[4]]).astype(np.float)
ar2=ar[3][0].replace("D","E",4).split(" ")
b=np.array([ar2[0],ar2[2],ar2[3],ar2[4]]).astype(np.float)

A1=5*10**-9
A2=np.array(list(map(lambda f:a[0]+a[1]*f+a[2]*f**2+a[3]*f**3,phiIPPm)))
A3=14
A4=np.array(list(map(lambda f:a[0]+b[1]*f+b[2]*f**2+b[3]*f**3,phiIPPm)))
t=(LongIPP*180/np.pi)/15+2
dT=np.array(list(map(lambda a2,a4,ti:A1+a2*cos(2*np.pi*(ti-A3)/a4),A2,A4,t)))

ZIonodelay=c*dT
IonodelayK=np.array(list(map(lambda delay, z: delay / cos(z), ZIonodelay, Ztag)))

print("עיכוב יונוספרי - בזניט ובכיוון הלווין")
MM.PrintMatrix(np.concatenate((ZIonodelay,IonodelayK),axis=1))


CoorInon, sIono = XYZcalculate(Observations[:, :, T], AproxPosition[:3].reshape((3, 1)), SetallitePosition, T_error * 10 ** -6, TDh, IonodelayK)
print("קרודינאטות לאחר התחשבות בעיכוב טרופוספרה ויונוספרה")
MM.PrintMatrix(np.concatenate((CoorInon,CoorInon-trueVal),axis=1))


CoorInonfree, sIonofree = XYZcalculate(Observations[:, :, T], AproxPosition[:3].reshape((3, 1)),SetallitePosition, T_error * 10 ** -6,TDh, IonoFree=True)
print("קרודינאטות לאחר התחשבות בעיכוב טרופוספרה וקומבינציית Iono-Free")
MM.PrintMatrix(np.concatenate((CoorInonfree,CoorInonfree-trueVal),axis=1))

##### Inosphera delay by IONEX ############################################################################
Hi=450 #hight of  Ionosphere in km

Longtitudei=LLH[1]*np.pi/180
Zniti=90-elev
Ztagi=np.array(list(map(lambda z: np.arcsin(Re/(Re+Hi)*sin(np.deg2rad(z))),Znit)))

print("Z and Ztag IONEX")
MM.PrintMatrix(np.concatenate((Zniti,np.rad2deg(Ztagi),Znit-np.rad2deg(Ztagi)),axis=1))


Dzi=np.deg2rad(Zniti)-Ztagi
Azi=np.array(list(map(lambda V:arctan2(V[1],V[2]),uen))).reshape((8,1))
phiIPPi=np.array(list(map(lambda dz,az: np.arcsin(sin(phi)*cos(dz)+cos(phi)*sin(dz)*cos(az)),Dzi,Azi)))
LongIPPi=np.array(list(map(lambda f,dz,az:Longtitude+np.arcsin((sin(dz)*sin(az))/f),phiIPPi,Dzi,Azi)))\

#calculte magnetic Position
phiIPPmi=np.array(list(map(lambda ff,ll:np.arcsin(sin(ff)*sin(phi0)+cos(ff)*cos(phi0)*cos(ll-L0)),phiIPPi,LongIPPi)))

print("Iono Position for each setalite - 450 KM")
MM.PrintMatrix(np.concatenate((np.rad2deg(phiIPPi),np.rad2deg(phiIPPmi),np.rad2deg(LongIPPi)),axis=1))

#### crate grid
lat = list(range(-875, 876, 25))
lat = [x / 10 for x in lat]
long = list(range(-180,180,5))

lat_cloest=list(map(lambda f: find_closest_index(lat,f),np.rad2deg(phiIPPmi)))
long_cloest=list(map(lambda l: find_closest_index(long,l),np.rad2deg(LongIPPi)))

lat1=np.array(list(map(lambda i: lat[lat_cloest[i]],range(len(lat_cloest)))))
long1=np.array(list(map(lambda i: long[long_cloest[i]],range(len(long_cloest)))))

lat2=[]
long2=[]
for i in range(len(lat1)):
    if lat1[i]-np.rad2deg(phiIPPmi[i])>0:
        lat2.append(lat1[i] - 2.5)
    else:
        lat2.append(lat1[i] + 2.5)
    if long1[i]-np.rad2deg(LongIPPi[i])>0:
        long2.append(long1[i] - 2.5)
    else:
        long2.append(long1[i] + 2.5)

#####################################################################################################
INOEXfile=pd.read_csv("CODG042018I.csv",header=None,skiprows=678).to_numpy()
info=INOEXfile[0:len(INOEXfile):429]
INOEXM=np.zeros((71,73,len(info)))
for k in range(len(info)):
    n=0
    for i in range(2,423,6):
        row = np.array([])
        for c in range(5):
            rowi = np.array(INOEXfile[k*429+i+c,0].split(" "))
            rowi = rowi[rowi!=''].astype(np.float)
            row = np.concatenate((row,rowi))
        INOEXM[n,:,k]=row
        n+=1
###########################################################################################


TECv=[]
for k in range(len(lat1)):
    E1=IONEXfinder(INOEXM,[lat1[k], long1[k]])
    E2=IONEXfinder(INOEXM,[lat1[k], long2[k]])
    E3=IONEXfinder(INOEXM,[lat2[k], long2[k]])
    E4=IONEXfinder(INOEXM,[lat2[k], long1[k]])
    TECv.append(np.average([E1,E2,E3,E4]))

ZIonodelayi=np.array(list(map(lambda T: 0.162*0.1*T, TECv)))
Ionodelayionex=np.array(list(map(lambda delay, z: delay / cos(z), ZIonodelayi, Ztag)))


CoorInon, sIono = XYZcalculate(Observations[:, :, T], AproxPosition[:3].reshape((3, 1)), SetallitePosition, T_error * 10 ** -6, TDh, Ionodelayionex)
print("קרודינאטות לאחר התחשבות בעיכוב טרופוספרה ויונוספרה IONEX")
MM.PrintMatrix(np.concatenate((CoorInon,CoorInon-trueVal),axis=1))


####### GeometryFree ########################################################################################################################
diono = XYZcalculate(Observations[:, :, T],GeomertyFree=True)


