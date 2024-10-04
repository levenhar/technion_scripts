from typing import List, Any

import matplotlib.pyplot as plt

import Reader
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import datetime
import scipy
from scipy.sparse.linalg import inv
import MatrixMethods as MM
import pyproj
from scipy.stats import norm

def Invers3x3(A):
    '''

    :param A: a matrix
    :type A: np.array 3x3
    :return: the inverse matrix. analytic calculation
    :rtype: np.array 3x3
    '''
    if A.shape[0] != 3 and A.shape[1] != 3 and len(A.shape) !=2:
        raise Exception("The metrix need to be 3x3")
    elif np.linalg.det(A) == 0:
        raise Exception("The metrix is singular")
    else:
        A = np.concatenate((np.array([[0, 0, 0]]),A))
        A = np.concatenate((np.array([[0 ,0, 0, 0]]).T, A), axis=1)
        a1 = [A[2, 2] * A[3, 3] - A[2, 3] * A[3, 2], A[2, 3] * A[3, 1] - A[2, 1] * A[3, 3], A[2, 1] * A[3, 2] - A[2, 2] * A[3, 1]]
        a2 = [A[1, 3] * A[3, 2] - A[1, 2] * A[3, 3], A[1, 1] * A[3, 3] - A[1, 3] * A[3, 1], A[1, 2] * A[3, 1] - A[1, 1] * A[3, 2]]
        a3 = [A[1, 2] * A[2, 3] - A[1, 3] * A[2, 2], A[1, 3] * A[2, 1] - A[1, 1] * A[2, 3], A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]]
        B = np.array([a1,a2,a3])
        return 1/np.linalg.det(A[1:,1:])*B.T


def InversBLockdiagonal3x3(A):
    '''

    :param A: block diagonal matrix. each block is 3x3
    :type A: np.array nxn
    :return: the inverse matrix
    :rtype: np.array nxn
    '''
    B = np.zeros(A.shape)
    for i in range(0,A.shape[0],3):
        B[i:i+3,i:i+3]=Invers3x3(A[i:i+3,i:i+3])
    return B



def Bootstrapping(vec, S):
    diff = abs(vec-np.round(vec))
    inx = np.argsort(diff,0).reshape(-1,)
    vecSorted = vec[inx,:]
    integerVec = np.zeros((len(vec)))
    vec = vecSorted.reshape(-1,)
    S1 = S[inx,:]
    S = S1[:, inx]
    integerVec[0] = round(vec[0])
    for i in range(1,len(vec)):
        si = 0
        for j in range(i-1,-1,-1):
            si += S[i,j]/S[j,j]*(vec[j]-integerVec[j])
        integerVec[i] = round(vec[i] - si)
    Reverseinx = np.argsort(inx)

    integerVec = integerVec[Reverseinx]
    return integerVec.reshape((len(integerVec),1))


def ILS(vec,S):
    integer_neighborhood = np.round(vec)
    GRID = np.zeros((4,len(integer_neighborhood)))
    for i in range(-2,2):
        GRID[i+2,:] = integer_neighborhood.reshape(len(integer_neighborhood))+i
    v=vec - integer_neighborhood
    val = (v.T@S@v)[0,0]
    integerval = np.zeros(14)
    for _ in range(100000):
        vector = np.random.randint(0, 3, 14)
        for i in range(14):
            integerval[i] = GRID[vector[i],i]
        v = vec - integerval.reshape((len(integerval),1))
        val1 = (v.T @ S @ v)[0, 0]
        if val1 < val:
            val = val1
            integer_neighborhood = integerval
    return integer_neighborhood



def find_closest_index(lst, value):
    """
    Returns the index of the element in lst that is closest to the given value.
    """
    return min(range(len(lst)), key=lambda m: abs(lst[m] - value))

def calculateSetllPosition(Rover,Base,Binfo,BsatelliteNum):
    SetllPListR = []
    SetllPListB = []
    for m in range(Rover.shape[2]):
        # for loop for each epoch
        RoverObs = Rover[:, :, m]
        BaseObs = Base[:, :, m]


        SetallitePositionR = []
        SetallitePositionB = []
        for i, prn in enumerate(BsatelliteNum[m]):
            # for loop for calculate the satellite position in the correct time
            if prn == "0.0":
                continue
            prn = "P" + prn
            indexx = np.where(Satell == prn)[0][0]
            prn_data = CoordinatesAll[indexx, :, :].T

            '''fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection = "3d")
            ax1.scatter(prn_data[:,0], prn_data[:,1], prn_data[:,2])
            plt.show()'''

            ttR = RoverObs[i, 0] / c
            ttB = BaseObs[i, 0] / c
            R_R = np.array([[np.cos(we * ttR), np.sin(we * ttR), 0], [-np.sin(we * ttR), np.cos(we * ttR), 0], [0, 0, 1]])
            R_B = np.array([[np.cos(we * ttB), np.sin(we * ttB), 0], [-np.sin(we * ttB), np.cos(we * ttB), 0], [0, 0, 1]])

            timee = Binfo[m].astype(np.float)

            data_timeR = datetime.datetime(int(timee[0]), int(timee[1]), int(timee[2]), int(timee[3]), int(timee[4]),
                                           int(timee[5])) - datetime.timedelta(seconds=ttR)
            data_timeR = np.float(data_timeR.timestamp())

            data_timeB = datetime.datetime(int(timee[0]), int(timee[1]), int(timee[2]), int(timee[3]), int(timee[4]),
                                           int(timee[5])) - datetime.timedelta(seconds=ttB)
            data_timeB = np.float(data_timeB.timestamp())

            j = find_closest_index(Timearray, data_timeR)

            #SP3Coordinate[cc, :] = prn_data[j, 1:4].astype(np.float)

            DATA4Poly = prn_data[j - 6:j + 7, :]
            DATA4Poly = DATA4Poly.astype(np.float)
            Times4Poly = Timearray[j - 6:j + 7]
            Times4Poly = Times4Poly.astype(np.float)

            coeffs = interp1d(Times4Poly, DATA4Poly.T, kind=3)
            try:
                pos_at_t_R = coeffs(data_timeR).T
            except:
                x=0
            pos_at_t_B = coeffs(data_timeB).T

            pos_at_t_R = R_R @ (pos_at_t_R[:3].reshape((3, 1)))
            SetallitePositionR.append(pos_at_t_R)

            pos_at_t_B = R_B @ (pos_at_t_B[:3].reshape((3, 1)))
            SetallitePositionB.append(pos_at_t_B)

            #MM.PrintMatrix(pos_at_t_R - pos_at_t_B)

        SetallitePositionR = np.array(SetallitePositionR)
        SetallitePositionR = SetallitePositionR.reshape(SetallitePositionR.shape[:2])
        SetallitePositionB = np.array(SetallitePositionB)
        SetallitePositionB = SetallitePositionB.reshape(SetallitePositionB.shape[:2])
        SetllPListR.append(SetallitePositionR)
        SetllPListB.append(SetallitePositionB)
    return SetllPListR, SetllPListB

CoordinatesAll, Information, Satell = Reader.readsp3("igu22141_00.sp3.txt")

ELAT = pd.read_csv("elat163v.22o.csv").to_numpy()
ELAT = np.array([row for row in ELAT if (row[0].startswith('G09') or row[0].startswith('G30') or row[0].startswith('G04') or row[0].startswith('G06') or row[0].startswith('>'))])

Base, BsatelliteNum, Binfo = [], [], []

s=0
OrbvsMatrix = []
for i in range(0, len(ELAT)-1):
    row = ELAT[i, 0].split(" ")
    dellist = []
    for j in range(len(row)):
        if row[j] == "":
            dellist.append(j)
    row = [element for index, element in enumerate(row) if index not in dellist]
    if row[0] == ">":
        s +=1
        Binfo.append(np.array([row[1:]]))
        if OrbvsMatrix != []:
            try:
                Base.append(np.array(OrbvsMatrix)[:, [1, 2, 9, 10, 17, 18]].astype(np.float).reshape((4, 6, 1)))
                BsatelliteNum.append(np.array(OrbvsMatrix)[:, 0].reshape((1, 4)))
            except:
                Binfo.pop()
                x = 0
        OrbvsMatrix = []
    else:


        OrbvsMatrix.append(row)
if OrbvsMatrix != []:
    try:
        Base.append(np.array(OrbvsMatrix)[:, [1, 2, 9, 10, 17, 18]].astype(np.float).reshape((4, 6, 1)))
        BsatelliteNum.append(np.array(OrbvsMatrix)[:, 0].reshape((1, 4)))
    except:
        Binfo.pop()
        x = 0

Base = np.concatenate(Base,axis=2)
BsatelliteNum = np.concatenate(BsatelliteNum)
Binfo  = np.concatenate(Binfo)

Base = Base[:,:,BsatelliteNum[:,0] == "G09"]
Binfo = Binfo[BsatelliteNum[:,0] == "G09",:]
BsatelliteNum = BsatelliteNum[BsatelliteNum[:,0] == "G09",:]


BSHM = pd.read_csv("bshm163v.22o.csv").to_numpy()
BSHM = np.array([row for row in BSHM if (row[0].startswith('G09') or row[0].startswith('G30') or row[0].startswith('G04') or row[0].startswith('G06') or row[0].startswith('>'))])
Rover, RsatelliteNum, Rinfo = [], [], []

# s=0
# for i in range(0, len(BSHM)-1,5):
#     row = BSHM[i-s, 0].split(" ")
#     if row[0] != ">":
#         s +=1
#         row = BSHM[i - s, 0].split(" ")
#     dellist = []
#     for j in range(len(row)):
#         if row[j] == "":
#             dellist.append(j)
#     row = [element for index, element in enumerate(row) if index not in dellist]
#
#     Rinfo.append(np.array([row[1:]]))
#     OrbvsMatrix = []
#     for k in range(1,5):
#         if i+k-s >= len(ELAT):
#             break
#         row = BSHM[i+k-s,0].split(" ")
#         dellist = []
#         for j in range(len(row)):
#             if row[j] == "":
#                 dellist.append(j)
#         row = [element for index, element in enumerate(row) if index not in dellist]
#         OrbvsMatrix.append(row)
#     try:
#         Rover.append(np.array(OrbvsMatrix)[:,[1,2,9,10,-4, -3]].astype(np.float).reshape((4,6,1)))
#         RsatelliteNum.append(np.array(OrbvsMatrix)[:,0].reshape((1,4)))
#     except:
#         Rinfo.pop()
#         x=0

OrbvsMatrix = []
for i in range(0, len(BSHM)-1):
    row = BSHM[i, 0].split(" ")
    dellist = []
    for j in range(len(row)):
        if row[j] == "":
            dellist.append(j)
    row = [element for index, element in enumerate(row) if index not in dellist]
    if row[0] == ">":
        Rinfo.append(np.array([row[1:]]))
        if OrbvsMatrix != []:
            try:
                Rover.append(np.array(OrbvsMatrix)[:, [1,2,9,10,-4, -3]].astype(np.float).reshape((4, 6, 1)))
                RsatelliteNum.append(np.array(OrbvsMatrix)[:, 0].reshape((1, 4)))
            except:
                Rinfo.pop()
                x = 0
        OrbvsMatrix = []
    else:
        OrbvsMatrix.append(row)
if OrbvsMatrix != []:
    try:
        Rover.append(np.array(OrbvsMatrix)[:, [1, 2, 9, 10, 17, 18]].astype(np.float).reshape((4, 6, 1)))
        RsatelliteNum.append(np.array(OrbvsMatrix)[:, 0].reshape((1, 4)))
    except:
        Rinfo.pop()
        x = 0


Rover = np.concatenate(Rover,axis=2)
RsatelliteNum = np.concatenate(RsatelliteNum)
Rinfo  = np.concatenate(Rinfo)

Rover = Rover[:,:,RsatelliteNum[:,0] == "G09"]
Rinfo = Rinfo[RsatelliteNum[:,0] == "G09",:]
RsatelliteNum = RsatelliteNum[RsatelliteNum[:,0] == "G09",:]


GoodIndexes =[]
for i in range(len(Rinfo)):
    matches = np.where(np.all(Rinfo[i,:-1].reshape((1,7)) == Binfo[:,:-1], axis=1))[0]
    if len(matches) > 0:
        GoodIndexes.append(i)

Rover = Rover[:,:,GoodIndexes]
Rinfo = Rinfo[GoodIndexes ,:]
RsatelliteNum = RsatelliteNum[GoodIndexes,:]


##### ELAT = Base
BasePosition = np.array([[4555028.3301, 3180067.5076, 3123164.667]]).T

#### BSMT = Rover
RAproxPosition = np.array([[4395951.4806, 3080707.0549, 3433498.0009]]).T

Timearray = [datetime.datetime(2022, 6, 12, 0, 0, 0)]
for i in range(1, 96):
    Timearray.append((Timearray[i - 1] + datetime.timedelta(minutes=15)))
for i in range(0, 96):
    Timearray[i] = np.float(Timearray[i].timestamp())
Timearray = np.array(Timearray).T

c = 299792458
we = (15 * np.pi / 180) / 3600  # rad/s
f1 = 1575.42*10**6
f2 = 1227.60*10**6
f5 = 1176.45*10**6
L1 = c/f1
L2 = c/f2
L5 = c/f5

Rover[:, 1, :] = L1 * Rover[:, 1, :]
Rover[:, 3, :] = L2 * Rover[:, 3, :]
Rover[:, 5, :] = L5 * Rover[:, 5, :]

Base[:, 1, :] = L1 * Base[:, 1, :]
Base[:, 3, :] = L2 * Base[:, 3, :]
Base[:, 5, :] = L5 * Base[:, 5, :]

u=0
dX = np.array([[np.inf, np.inf, np.inf]]).T

SPositionR, SPositionB = calculateSetllPosition(Rover, Base, Binfo[:,:-2], BsatelliteNum)

for i in range(len(SPositionR)):
    SPositionR[i] = SPositionR[i]*10**3
    SPositionB[i] = SPositionB[i]*10**3

while np.linalg.norm(dX) > 0.0001:
    L = []
    P = []
    A = []
    for m in range(72):  #Base.shape[2]
        # for loop for each epoch
        t = Rinfo[m][:6]
        RoverObs = Rover[:, :, m]
        SatR = RsatelliteNum[m]

        BaseObs = Base[:, :, m]
        SatB = BsatelliteNum[m]
        NumofN = 9
        ############# Linear Combination ###################################################################
        if False:
            RoverObs = (4*f1*RoverObs[:,:2] -f2*RoverObs[:,2:4] -2*f5*RoverObs[:,4:])/(4*f1 -f2 -2*f5)
            BaseObs = (4*f1*BaseObs[:,:2] -f2*BaseObs[:,2:4] -2*f5*BaseObs[:,4:])/(4*f1 -f2 -2*f5)
            L1 = 0.1102
            NumofN=3

        SetallitePositionR = SPositionR[m]
        SetallitePositionB = SPositionB[m]

        aproxRangeR = np.zeros(SetallitePositionR.shape[0])
        aproxRangeB = np.zeros(SetallitePositionB.shape[0])
        Atag = np.zeros(SetallitePositionB.shape)
        for n in range(len(SetallitePositionR)):
            # calculate the distance from the stations to the satellites
            vR = SetallitePositionR[n, :].reshape((3, 1)) - RAproxPosition
            vB = SetallitePositionB[n, :].reshape((3, 1)) - BasePosition

            aproxRangeR[n] = np.sqrt(vR.T @ vR)[0][0]
            aproxRangeB[n] = np.sqrt(vB.T @ vB)[0][0]
            Atag[n,:] = (vR/aproxRangeR[n]).reshape(-1,) #MEKDEM for one setallite

        A1 = Atag[1:,:]-Atag[0,:] # calculate the MEKADMIM for position varible
        At = []
        SingleDiff = RoverObs - BaseObs
        DoubleDiff = SingleDiff[1:, :] - SingleDiff[0, :]
        for l in range(len(RoverObs[0])):
            A2 = np.zeros((3, NumofN))
            for i in range(1,len(RoverObs)):
                if RoverObs[i,l] == 0 or BaseObs[i,l] == 0:
                    continue
                else:
                    li = DoubleDiff[i-1,l] - aproxRangeB[0] + aproxRangeB[i] + aproxRangeR[0] - aproxRangeR[i]
                    if abs(li)>100:
                        x=0
                    L.append(-li)
                    if l==1:
                        A2[i-1,i-1] = L1
                        P.append(1 / 0.003**2)
                    elif l==3:
                        A2[i - 1, i +2] = L2
                        P.append(1 / 0.003**2)
                    elif l == 5:
                        A2[i - 1, i + 5] = L5
                        P.append(1 / 0.003 ** 2)
                    else:
                        P.append(1 / 0.3**2)
            Ai = np.concatenate((A1,A2),axis=1)
            At.append(Ai)
        At = np.concatenate(At)
        A.append(At)

    A = np.concatenate(A)
    A = scipy.sparse.csc_matrix(A)
    P = scipy.sparse.dia_matrix(([P], [0]), shape=(len(P), len(P)))
    L = scipy.sparse.csc_matrix(L).T

    N = (A.T@P@A).tocsc()
    U = (A.T@P@L).tocsc()

    X = inv(N)@U
    dX = X[:3].toarray()
    RAproxPosition += dX
    u+=1

print("Float Solution")
V = A @ X - L

s = np.sqrt(((V.T @ P @ V) / (A.shape[0] - A.shape[1]))[0, 0])
Sx = s ** 2 * inv(N)
#MM.PrintMatrix(np.concatenate((X[3:].toarray(),np.sqrt(np.diag(Sx[3:,3:].toarray())).reshape((9,1))),axis = 1))
MM.PrintMatrix(np.sqrt(np.diag(Sx.toarray())))
floatSolution = np.copy(RAproxPosition)
MM.PrintMatrix(RAproxPosition)

TrueSolution = np.array([[4395951.1938, 3080707.2235, 3433498.2537]]).T
MM.PrintMatrix(TrueSolution - floatSolution)

# Define the geocentric coordinate system (XYZ)
geocentric = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
# Define the UTM coordinate system
utm = pyproj.Proj(proj='utm', zone=36, ellps='WGS84', datum='WGS84')
transformer = pyproj.Transformer.from_proj(geocentric, utm, always_xy=True)


easting1, northing1, h1 = transformer.transform(TrueSolution[0,0], TrueSolution[1, 0], TrueSolution[2, 0])
easting2, northing2, h2 = transformer.transform(floatSolution[0, 0], floatSolution[1, 0], floatSolution[2, 0])

print(easting1 - easting2)
print(northing1 - northing2)
print(h1 - h2)

BaseVector = np.linalg.norm(TrueSolution - BasePosition)
print(f"The Baseline length is {BaseVector}")