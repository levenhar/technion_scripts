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

def calculateSetllPosition(LimitsR,LimitsB,Rover,Rinfo,RsatelliteNum,Base,BsatelliteNum):
    SetllPListR = []
    SetllPListB = []
    for m in range(LimitsR[1] - LimitsR[0]):
        # for loop for each epoch
        RoverObs = Rover[:, :, LimitsR[0] + m]
        BaseObs = Base[:, :, LimitsB[0] + m]


        SetallitePositionR = []
        SetallitePositionB = []
        for i, prn in enumerate(BsatelliteNum[LimitsB[0] + m]):
            # for loop for calculate the satellite position in the correct time
            if prn == "0.0":
                continue
            prn = "PG" + prn
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

            timee = Binfo[LimitsB[0] + m].astype(np.float)

            data_timeR = datetime.datetime(int(2000 + timee[0]), int(timee[1]), int(timee[2]), int(timee[3]), int(timee[4]),
                                           int(timee[5])) - datetime.timedelta(seconds=ttR)
            data_timeR = np.float(data_timeR.timestamp())

            data_timeB = datetime.datetime(int(2000 + timee[0]), int(timee[1]), int(timee[2]), int(timee[3]), int(timee[4]),
                                           int(timee[5])) - datetime.timedelta(seconds=ttB)
            data_timeB = np.float(data_timeB.timestamp())

            j = find_closest_index(Timearray, data_timeR)

            # SP3Coordinate[cc, :] = prn_data[j, 1:4].astype(np.float)

            DATA4Poly = prn_data[j - 6:j + 7, :]
            DATA4Poly = DATA4Poly.astype(np.float)
            Times4Poly = Timearray[j - 6:j + 7]
            Times4Poly = Times4Poly.astype(np.float)

            coeffs = interp1d(Times4Poly, DATA4Poly.T, kind=3)
            pos_at_t_R = coeffs(data_timeR).T
            pos_at_t_B = coeffs(data_timeB).T

            pos_at_t_R = R_R @ (pos_at_t_R[:3].reshape((3, 1)))
            SetallitePositionR.append(pos_at_t_R)

            pos_at_t_B = R_B @ (pos_at_t_B[:3].reshape((3, 1)))
            SetallitePositionB.append(pos_at_t_B)
        SetallitePositionR = np.array(SetallitePositionR)
        SetallitePositionR = SetallitePositionR.reshape(SetallitePositionR.shape[:2])
        SetallitePositionB = np.array(SetallitePositionB)
        SetallitePositionB = SetallitePositionB.reshape(SetallitePositionB.shape[:2])
        SetllPListR.append(SetallitePositionR)
        SetllPListB.append(SetallitePositionB)
    return SetllPListR, SetllPListB

CoordinatesAll, Information, Satell = Reader.readsp3("igs22113sp3.txt")
Rover, RsatelliteNum, Rinfo, RAproxPosition = Reader.readOfile("Rov_log0145a22o.txt")
Base, BsatelliteNum, Binfo, _ = Reader.readOfile("Base_log0145a22o.txt")


BasePosition = np.array([[4475707.198, 3087483.756, 3323084.834]]).T
RAproxPosition = RAproxPosition.reshape((3, 1))


# Define the geocentric coordinate system (XYZ)
geocentric = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
# Define the UTM coordinate system
utm = pyproj.Proj(proj='utm', zone=36, ellps='WGS84', datum='WGS84')
transformer = pyproj.Transformer.from_proj(geocentric, utm, always_xy=True)

RlimitsStati = [0, 0]
RlimitsKinimati = [0, 0]

Timearray = [datetime.datetime(2022, 5, 25, 0, 0, 0)]
for i in range(1, 96):
    Timearray.append((Timearray[i - 1] + datetime.timedelta(minutes=15)))
for i in range(0, 96):
    Timearray[i] = np.float(Timearray[i].timestamp())
Timearray = np.array(Timearray).T

for i in range(len(Rinfo)):
    if all(Rinfo[i][:6] == np.array(["22", "5", "25", "12", "2", "0.0000000"])):
        RlimitsStati[0] = i
    elif all(Rinfo[i][:6] == np.array(["22", "5", "25", "12", "11", "0.0000000"])):
        RlimitsStati[1] = i
        RlimitsKinimati[0] = i + 1
    elif all(Rinfo[i][:6] == np.array(["22", "5", "25", "12", "20", "0.0000000"])):
        RlimitsKinimati[1] = i
        break
BlimitsStati = [0, 0]
BlimitsKinimati = [0, 0]
for i in range(len(Binfo)):
    if all(Binfo[i][:6] == np.array(["22", "5", "25", "12", "2", "0.0000000"])):
        BlimitsStati[0] = i
    elif all(Binfo[i][:6] == np.array(["22", "5", "25", "12", "11", "0.0000000"])):
        BlimitsStati[1] = i
        BlimitsKinimati[0] = i + 1
    elif all(Binfo[i][:6] == np.array(["22", "5", "25", "12", "20", "0.0000000"])):
        BlimitsKinimati[1] = i
        break
test = ["10", "15", "16", "18", "23", "26", "27", "29"]

c = 299792458
we = (15 * np.pi / 180) / 3600  # rad/s
f1 = 1575.42*10**6
f2 = 1227.60*10**6
L1 = c/f1
L2 = c/f2

Rover[:, 1, :] = L1 * Rover[:, 1, :]
Rover[:, 2, :] = L2 * Rover[:, 2, :]

Base[:, 1, :] = L1 * Base[:, 1, :]
Base[:, 2, :] = L2 * Base[:, 2, :]


u=0
dX = np.array([[np.inf, np.inf, np.inf]]).T

SPositionR, SPositionB = calculateSetllPosition(RlimitsStati, BlimitsStati, Rover, Rinfo, RsatelliteNum, Base, BsatelliteNum)

for i in range(len(SPositionR)):
    SPositionR[i] = SPositionR[i]*10**3
    SPositionB[i] = SPositionB[i]*10**3

while True:
    L: List[Any] = []
    P = []
    A = []
    for m in range(RlimitsStati[1] - RlimitsStati[0]):
        # for loop for each epoch
        t = Rinfo[RlimitsStati[0] + m][:6]
        RoverObs = Rover[:, :, RlimitsStati[0] + m]
        SatR = RsatelliteNum[RlimitsStati[0] + m]

        BaseObs = Base[:, :, BlimitsStati[0] + m]
        SatB = BsatelliteNum[BlimitsStati[0] + m]

        SetallitePositionR = SPositionR[m]
        SetallitePositionB = SPositionB[m]

        #diff = SetallitePositionR - SetallitePositionB

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
        SingleDiff = RoverObs[:8,:] - BaseObs[:8,:]
        DoubleDiff = SingleDiff[1:, :] - SingleDiff[0, :]
        for l in range(len(RoverObs[0])):
            A2 = np.zeros((7, 14))
            for i in range(1,len(RoverObs)):
                if RoverObs[i,l] == 0 or BaseObs[i,l] == 0:
                    continue
                else:
                    li = DoubleDiff[i-1,l] - aproxRangeB[0] + aproxRangeB[i] + aproxRangeR[0] - aproxRangeR[i]
                    L.append(-li)
                    if l==1:
                        A2[i-1,i-1] = L1
                        P.append(1 / 0.003**2)
                    elif l==2:
                        A2[i - 1, i +6] = L2  #for seperated ambugity varible change -1 to +6 in the colounm
                        P.append(1 / 0.003**2)
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
    '''MM.PrintMatrix((X[3:].toarray()).T)
    print(u)'''
    u+=1
    if np.linalg.norm(dX) < 0.0001:
        print("Float Solution")
        V = A @ X - L

        s = np.sqrt(((V.T @ P @ V) / (A.shape[0] - A.shape[1]))[0, 0])
        Sx = s ** 2 * inv(N)
        MM.PrintMatrix(np.concatenate((X[3:].toarray(),np.sqrt(np.diag(Sx[3:,3:].toarray())).reshape((14,1))),axis = 1))
        MM.PrintMatrix(np.sqrt(np.diag(Sx.toarray())))
        floatSolution = np.copy(RAproxPosition)
        MM.PrintMatrix(RAproxPosition)
        TopconSolution = np.array([[4475707.676, 3087477.470, 3323088.155]]).T

        vv = RAproxPosition - TopconSolution
        MM.PrintMatrix(vv)

        print(np.sqrt(vv.T @ vv)[0,0])

        dX = np.array([[np.inf, np.inf, np.inf]]).T
        aproxPosition = X[:3].toarray()
        #IntegerAbugity = Bootstrapping(X[3:].toarray(),Sx.toarray()[3:,3:])
        #IntegerAbugity = np.round(X[3:].toarray())
        #IntegerAbugity = ILS(X[3:].toarray(),Sx[3:,3:].toarray())
        IntegerAbugity = np.array([[30,22,-54,67,-2,-23,-95,30,21,-56,67,-2,-24,-100]]).T

        Pb = 1
        mean = 0
        std_dev = 1
        SXX = Sx.toarray()
        for i in range(len(IntegerAbugity)):
            Pb = Pb*(2*norm.cdf( 1/(2*np.sqrt(SXX[i+3,i+3])),mean, std_dev)-1)
        #Pb = Pb-1
        print(f"the FIX probability is {Pb}")


        while np.linalg.norm(dX) > 0.0001:
            L = []
            P = []
            A = []
            for m in range(RlimitsStati[1] - RlimitsStati[0]):
                # for loop for each epoch
                t = Rinfo[RlimitsStati[0] + m][:6]
                RoverObs = Rover[:, :, RlimitsStati[0] + m]
                SatR = RsatelliteNum[RlimitsStati[0] + m]

                BaseObs = Base[:, :, BlimitsStati[0] + m]
                SatB = BsatelliteNum[BlimitsStati[0] + m]

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
                    Atag[n, :] = (vR / aproxRangeR[n]).reshape(-1, )  # MEKDEM for one setallite

                A1 = Atag[1:, :] - Atag[0, :]  # calculate the MEKADMIM for position varible
                At = []
                SingleDiff = RoverObs[:8, :] - BaseObs[:8, :]
                DoubleDiff = SingleDiff[1:, :] - SingleDiff[0, :]
                for l in range(len(RoverObs[0])):
                    for i in range(1, len(RoverObs)):
                        if RoverObs[i, l] == 0 or BaseObs[i, l] == 0:
                            continue
                        else:
                            li = DoubleDiff[i - 1, l] - aproxRangeB[0] + aproxRangeB[i] + aproxRangeR[0] - aproxRangeR[i]
                            if l == 1:
                                li += L1*IntegerAbugity[i-1,0]
                                P.append(1 / 0.003**2)
                            elif l == 2:
                                P.append(1 / 0.003**2)
                                li += L2 * IntegerAbugity[i +6, 0] #for seperated ambugity varible change -1 to +6 in the colounm
                            else:
                                P.append(1 / 0.3**2)
                            L.append(-li)
                    At.append(A1)
                At = np.concatenate(At)
                A.append(At)
            A = np.concatenate(A)
            #A = scipy.sparse.csc_matrix(A)
            P = scipy.sparse.dia_matrix(([P], [0]), shape=(len(P), len(P)))
            #L = scipy.sparse.csc_matrix(L).T
            L = np.array(L).reshape((len(L),1))

            N = A.T @ P @ A
            U = A.T @ P @ L

            dX = np.linalg.inv(N) @ U
            #dX = X[:3].toarray()
            RAproxPosition += dX
        break

print("Fix Solution")
V = A@dX-L

s = np.sqrt(((V.T@P@V)/(A.shape[0]-A.shape[1]))[0,0])
Sx = s**2*np.linalg.inv(N)

MM.PrintMatrix(np.sqrt(np.diag(Sx)))


MM.PrintMatrix(RAproxPosition)
TopconSolution = np.array([[4475707.676, 3087477.470, 3323088.155]]).T

vv = RAproxPosition - TopconSolution
MM.PrintMatrix(vv)

print(np.sqrt(vv.T @ vv)[0,0])

print("")
print("diff between Float and Fix")
MM.PrintMatrix(floatSolution - RAproxPosition)


##### Kinematic solution ########################################################################################################################
KPositionR, KPositionB = calculateSetllPosition(RlimitsKinimati, BlimitsKinimati, Rover, Rinfo, RsatelliteNum, Base, BsatelliteNum)

for i in range(len(KPositionR)):
    KPositionR[i] = KPositionR[i]*10**3
    KPositionB[i] = KPositionB[i]*10**3

dX = np.array([[np.inf, np.inf, np.inf]]).T
RAproxPosition = [RAproxPosition]
for m in range(len(KPositionR)-1):
    RAproxPosition.append(RAproxPosition[-1])

RAproxPosition = np.concatenate(RAproxPosition,axis=1).T
RAproxPositionbu = np.copy(RAproxPosition)



while max(abs(dX.reshape(-1,)))>0.001:
    L = []
    P = []
    Alist = []

    for m in range(len(KPositionR)):
        t = Rinfo[RlimitsKinimati[0] + m][:6]
        RoverObs = Rover[:, :, RlimitsKinimati[0] + m]
        SatR = RsatelliteNum[RlimitsKinimati[0] + m]

        BaseObs = Base[:, :, BlimitsKinimati[0] + m]
        SatB = BsatelliteNum[BlimitsKinimati[0] + m]

        SetallitePositionR = KPositionR[m]
        SetallitePositionB = KPositionB[m]

        '''if all(SatB[:8] != test) or all(SatB[:8] != test) :
            print("fuck")
            x=0'''

        aproxRangeR = np.zeros(SetallitePositionR.shape[0])
        aproxRangeB = np.zeros(SetallitePositionB.shape[0])
        Atag = np.zeros(SetallitePositionB.shape)
        for n in range(len(SetallitePositionR)):
            # calculate the distance from the stations to the satellites
            vR = SetallitePositionR[n, :].reshape((3, 1)) - RAproxPosition[m,:].reshape((3, 1))
            vB = SetallitePositionB[n, :].reshape((3, 1)) - BasePosition

            aproxRangeR[n] = np.sqrt(vR.T @ vR)[0][0]
            aproxRangeB[n] = np.sqrt(vB.T @ vB)[0][0]
            Atag[n, :] = (vR / aproxRangeR[n]).reshape(-1, )  # MEKDEM for one setallite

        A1 = Atag[1:, :] - Atag[0, :]  # calculate the MEKADMIM for position varible
        At = []
        SingleDiff = RoverObs[:8, :] - BaseObs[:8, :]
        DoubleDiff = SingleDiff[1:, :] - SingleDiff[0, :]
        for l in range(len(RoverObs[0])):
            for i in range(1, len(RoverObs)):
                if RoverObs[i, l] == 0 or BaseObs[i, l] == 0:
                    continue
                else:
                    li = DoubleDiff[i - 1, l] - aproxRangeB[0] + aproxRangeB[i] + aproxRangeR[0] - aproxRangeR[i]
                    if l == 1:
                        li += L1*IntegerAbugity[i-1,0]
                        P.append(1 / 0.003**2)
                    elif l == 2:
                        P.append(1 / 0.003**2)
                        li += L2 * IntegerAbugity[i +6, 0] #for seperated ambugity varible change -1 to +6 in the colounm
                    else:
                        P.append(1 / 0.3**2)
                    L.append(-li)
            At.append(A1)
        At = np.concatenate(At)
        Alist.append(At)

    sparse_matrices = [scipy.sparse.dia_matrix(array) for array in Alist]
    diagonal_matrix = scipy.sparse.block_diag(sparse_matrices)
    A = diagonal_matrix.tobsr()

    P = scipy.sparse.dia_matrix(([P], [0]), shape=(len(P), len(P)))
    L = np.array(L).reshape((len(L), 1))

    N = A.T @ P @ A
    U = A.T @ P @ L

    dX = InversBLockdiagonal3x3(N.toarray())@U
    #dX = inv(N)@U
    dX = dX.reshape((len(RAproxPosition),3))
    RAproxPosition = RAproxPosition+dX

easting1, northing1, h1 = transformer.transform(RAproxPosition[:, 0], RAproxPosition[:, 1], RAproxPosition[:, 2])

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(easting1, northing1, h1)
FixSolution = np.copy(RAproxPosition)

##### Float Kinamatic Solution ###########################################################################################################
dX = np.array([[np.inf, np.inf, np.inf]]).T


RAproxPosition = np.copy(RAproxPositionbu)
IntegerAbugity = X[3:]

while np.linalg.norm(dX.reshape(-1, )) > 0.0001:
    L = []
    P = []
    Alist = []

    for m in range(len(KPositionR)):
        t = Rinfo[RlimitsKinimati[0] + m][:6]
        RoverObs = Rover[:, :, RlimitsKinimati[0] + m]
        SatR = RsatelliteNum[RlimitsKinimati[0] + m]

        BaseObs = Base[:, :, BlimitsKinimati[0] + m]
        SatB = BsatelliteNum[BlimitsKinimati[0] + m]

        SetallitePositionR = KPositionR[m]
        SetallitePositionB = KPositionB[m]

        '''if all(SatB[:8] != test) or all(SatB[:8] != test) :
            print("fuck")
            x=0'''

        aproxRangeR = np.zeros(SetallitePositionR.shape[0])
        aproxRangeB = np.zeros(SetallitePositionB.shape[0])
        Atag = np.zeros(SetallitePositionB.shape)
        for n in range(len(SetallitePositionR)):
            # calculate the distance from the stations to the satellites
            vR = SetallitePositionR[n, :].reshape((3, 1)) - RAproxPosition[m, :].reshape((3, 1))
            vB = SetallitePositionB[n, :].reshape((3, 1)) - BasePosition

            aproxRangeR[n] = np.sqrt(vR.T @ vR)[0][0]
            aproxRangeB[n] = np.sqrt(vB.T @ vB)[0][0]
            Atag[n, :] = (vR / aproxRangeR[n]).reshape(-1, )  # MEKDEM for one setallite

        A1 = Atag[1:, :] - Atag[0, :]  # calculate the MEKADMIM for position varible
        At = []
        SingleDiff = RoverObs[:8, :] - BaseObs[:8, :]
        DoubleDiff = SingleDiff[1:, :] - SingleDiff[0, :]
        for l in range(len(RoverObs[0])):
            for i in range(1, len(RoverObs)):
                if RoverObs[i, l] == 0 or BaseObs[i, l] == 0:
                    continue
                else:
                    li = DoubleDiff[i - 1, l] - aproxRangeB[0] + aproxRangeB[i] + aproxRangeR[0] - aproxRangeR[i]
                    if l == 1:
                        li += L1 * IntegerAbugity[i - 1, 0]
                        P.append(1 / 0.003)
                    elif l == 2:
                        P.append(1 / 0.003)
                        li += L2 * IntegerAbugity[
                            i + 6, 0]  # for seperated ambugity varible change -1 to +6 in the colounm
                    else:
                        P.append(1 / 0.3)
                    L.append(-li)
            At.append(A1)
        At = np.concatenate(At)
        Alist.append(At)

    sparse_matrices = [scipy.sparse.dia_matrix(array) for array in Alist]
    diagonal_matrix = scipy.sparse.block_diag(sparse_matrices)
    A = diagonal_matrix.tobsr()

    P = scipy.sparse.dia_matrix(([P], [0]), shape=(len(P), len(P)))
    L = np.array(L).reshape((len(L), 1))

    N = A.T @ P @ A
    U = A.T @ P @ L

    dX = InversBLockdiagonal3x3(N.toarray()) @ U
    dX = dX.reshape((len(RAproxPosition), 3))
    RAproxPosition = RAproxPosition + dX

easting2, northing2, h2 = transformer.transform(RAproxPosition[:, 0], RAproxPosition[:, 1], RAproxPosition[:, 2])

#ax.scatter(easting2, northing2, h2,c = "red")
plt.show()

#MM.PrintMatrix(np.mean(FixSolution-RAproxPosition, 0))


# Define the value and parameters of the normal distribution
value = 1.96
mean = 0
std_dev = 1

# Calculate the cumulative distribution function (CDF)
cdf = norm.cdf(value, mean, std_dev)
x = 9
