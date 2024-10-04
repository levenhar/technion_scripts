from time import time
import matplotlib.pyplot as plt
import numpy as np
from SyntheticObject import syntheticImages
import copy
from SingleImage import SingleImage
from Camera import Camera
from Point2D import ImagePoints
from Point3D import GroundPoint
import scipy.sparse as sc
import MatrixMethods as MM


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


def computeA(GroundTiePoints,IDsGT,GCPlist,Shapee,ImageTiePointsList,ImageColntrolPointsList):
    '''
    function that calculat A matrix and L0 vector in thay full form
    :param GroundTiePoints: list of all the Tie points
    :type GroundTiePoints: np.array()
    :param IDsGT:ID list of the tie points
    :type IDsGT:np.array()
    :param GCPlist:list of the GCP coordinates
    :type GCPlist:list of GroundPoints
    :param Shapee:the shape of the Ground tie points matrix
    :type Shapee: tuple
    :param ImageTiePointsList: lists of the Image coordinate for the tie points
    :type ImageTiePointsList: list of arrays of ImagePoints
    :param ImageColntrolPointsList:lists of the Image coordinate for the control points
    :type ImageColntrolPointsList: list of arrays of ImagePoints
    :return: A, L0
    :rtype: np.array
    '''
    GroundControlPoints = GroundPoint.listofPoints2Array(GCPlist)
    IDsG = GroundControlPoints[:, 0]
    GroundControlPoints = GroundControlPoints[:, 1:].astype(np.float)

    GroundTiePointsM = copy.deepcopy(GroundTiePoints).reshape(Shapee)
    Alist = []
    Blist = []
    L0List= []
    for i in range(len(ImageList)):
        ###### order Control points #######################################
        ImageControlPoints=ImagePoints.listofPoints2Array(ImageColntrolPointsList[i])
        IDCs=ImageControlPoints[:,0]
        ImageControlPoints = ImageControlPoints[:,1:].astype(np.float)

        GCP = np.zeros((len(ImageControlPoints),3))
        for j in range(len(ImageControlPoints)):
            for id in range(len(IDsG)):
                if IDCs[j] == IDsG[id]:
                    GCP[j,:] = GroundControlPoints[id,:]

        ###### order Tie points #######################################
        ImageTiePoints = ImagePoints.listofPoints2Array(ImageTiePointsList[i])
        IDs = ImageTiePoints[:, 0]
        ImageTiePoints = ImageTiePoints[:, 1:].astype(np.float)

        TCP = np.zeros((len(ImageTiePoints), 3))
        for j in range(len(ImageTiePoints)):
            for id in range(len(IDsGT)):
                if IDs[j] == IDsGT[id]:
                    TCP[j, :] = GroundTiePointsM[id, :]

        Acp=ImageList[i].ComputeDesignMatrix(GCP)
        Atp=ImageList[i].ComputeDesignMatrix(TCP)

        Lcp=ImageList[i].ComputeObservationVector(GCP)
        Ltp=ImageList[i].ComputeObservationVector(TCP)

        Bcp=np.zeros((len(Acp),3*len(GroundTiePointsM)))
        Btp=np.zeros((len(Atp),3*len(GroundTiePointsM)))

        c=0
        for i in range(0,len(Atp),2):
            Btp[i:i+2,c:c+3]=-Atp[i:i+2,0:3]
            c+=3
        Ai=np.concatenate((Acp,Atp))
        Bi=np.concatenate((Bcp,Btp))
        L0i=np.concatenate((Lcp,Ltp))
        Alist.append(Ai)
        Blist.append(Bi)
        L0List.append(L0i)

    m=0
    n=0
    for k in range(len(Alist)):
        m += Alist[k].shape[1]
        n += Alist[k].shape[0]

    A=np.zeros((n,m))
    offsetR=0
    offsetC=0
    for k in range(len(Alist)):
        A[offsetR:offsetR+Alist[k].shape[0], offsetC:offsetC+Alist[k].shape[1]] = Alist[k]
        offsetR += Alist[k].shape[0]
        offsetC += Alist[k].shape[1]

    B=np.concatenate(Blist)
    A=np.concatenate((A,B),axis=1)
    L0=np.concatenate(L0List)

    return A, L0

def computeLb(ImageTiePointsList,ImageColntrolPointsList):
    '''
    function that calculate Lb vector
    :param ImageTiePointsList: lists of the Image coordinate for the tie points
    :type ImageTiePointsList: list of arrays of ImagePoints
    :param ImageColntrolPointsList:lists of the Image coordinate for the control points
    :type ImageColntrolPointsList: list of arrays of ImagePoints
    :return: Lb
    :rtype: np.array nx1
    '''
    LbList=[]
    for i in range(len(ImageTiePointsList)):
        ITP = ImagePoints.listofPoints2Array(ImageTiePointsList[i])[:, 1:].astype(np.float)
        ICP = ImagePoints.listofPoints2Array(ImageColntrolPointsList[i])[:, 1:].astype(np.float)
        Lbi=np.concatenate((ICP.reshape(-1,),ITP.reshape(-1,)))
        LbList.append(Lbi)

    Lb = np.concatenate(LbList)
    return Lb.reshape((len(Lb),1))


def ComputeNU(GroundTiePoints,IDsGT,GCPlist,Shapee,ImageTiePointsList,ImageColntrolPointsList, lambdaa, M, dlLast):
    '''
    function that calculat N matrix and u vector in directly, according to A blocks
    :param GroundTiePoints: list of all the Tie points
    :type GroundTiePoints: np.array()
    :param IDsGT:ID list of the tie points
    :type IDsGT:np.array()
    :param GCPlist:list of the GCP coordinates
    :type GCPlist:list of GroundPoints
    :param Shapee:the shape of the Ground tie points matrix
    :type Shapee: tuple
    :param ImageTiePointsList: lists of the Image coordinate for the tie points
    :type ImageTiePointsList: list of arrays of ImagePoints
    :param ImageColntrolPointsList:lists of the Image coordinate for the control points
    :type ImageColntrolPointsList: list of arrays of ImagePoints
    :return: N11, N22inverse, U1, U2, N12, N22
    :rtype: np.array / bsr_matrix
    '''
    GroundControlPoints = GroundPoint.listofPoints2Array(GCPlist)
    IDsG = GroundControlPoints[:, 0]
    GroundControlPoints = GroundControlPoints[:, 1:].astype(np.float)

    GroundTiePointsM = copy.deepcopy(GroundTiePoints).reshape(Shapee)
    Alist = []
    Blist = []
    LList = []
    LtList = []
    LcList = []
    TieindexList = []
    for i in range(len(ImageList)):
        ###### order Control points #######################################
        ImageControlPoints = ImagePoints.listofPoints2Array(ImageColntrolPointsList[i])
        IDCs = ImageControlPoints[:, 0]
        ImageControlPoints = ImageControlPoints[:, 1:].astype(np.float)

        rmovelist = []
        for j in range(len(ImageControlPoints)):
            if any(np.isnan(ImageControlPoints[j])):
                rmovelist.append(j)
        IDCs = np.delete(IDCs, rmovelist)

        GCP = np.delete(GroundControlPoints,rmovelist,axis=0)

        ICP = np.delete(ImageControlPoints,rmovelist,axis=0)

        ###### order Tie points #######################################
        ImageTiePoints = ImagePoints.listofPoints2Array(ImageTiePointsList[i])
        IDs = ImageTiePoints[:, 0]
        ImageTiePoints = ImageTiePoints[:, 1:].astype(np.float)

        rmovelist = []
        for j in range(len(ImageTiePoints)):
            if any(np.isnan(ImageTiePoints[j])):
                rmovelist.append(j)

        ITP = np.delete(ImageTiePoints, rmovelist, axis=0)
        IDs = np.delete(IDs, rmovelist)
        TieindexList.append(IDs)

        TCP = np.delete(GroundTiePointsM,rmovelist,axis=0)

        Acp = ImageList[i].ComputeDesignMatrix(GCP)
        Atp = ImageList[i].ComputeDesignMatrix(TCP)

        Lcp = ImageList[i].ComputeObservationVector(GCP)
        Ltp = ImageList[i].ComputeObservationVector(TCP)

        Ai = np.concatenate((Acp, Atp))
        Alist.append(Ai)
        Blist.append(-Atp[:,0:3])

        #ICP = ImagePoints.listofPoints2Array(ImageColntrolPointsList[i])[:, 1:].astype(np.float)
        #ITP = ImagePoints.listofPoints2Array(ImageTiePointsList[i])[:, 1:].astype(np.float)

        Lc = ICP.reshape(-1,) - Lcp
        Lt = ITP.reshape(-1,) - Ltp

        Li = np.concatenate((Lc, Lt))
        LList.append(Li)
        LtList.append(Lt)
        LcList.append(Lc)

    N11list = []
    for i in range(len(Alist)):
        N = Alist[i].T@Alist[i]
        N11list.append(N)

    sparse_matrices = [sc.dia_matrix(array) for array in N11list]
    diagonal_matrix = sc.block_diag(sparse_matrices)
    N11 = diagonal_matrix.tobsr()

    N22BlockList = [np.zeros((3,3)) for _ in range(len(GroundTiePointsM))]
    N12 = np.zeros((6*len(ImageList),3*len(GroundTiePointsM)))
    U = np.zeros((6*len(ImageList)+3*len(GroundTiePointsM),1))
    for i in range(len(Alist)):
        U[6*i:6*i+6,:] = (Alist[i].T@LList[i]).reshape((6,1))

    c=0
    for i in range(len(GroundTiePointsM)):
        GTPid = IDsGT[i]
        for b in range(len(Blist)):
            for k in range(len(TieindexList[b])):
                ITPid = TieindexList[b][k]
                if GTPid == ITPid:
                    ll = LtList[b][2 * k:2 * k + 2].reshape((2, 1))
                    aindex = np.where(IDsGT == ITPid)[0][0]
                    bb = Blist[b][2 * k:2 * k + 2, :]
                    N22BlockList[i] += bb.T @ bb
                    offset = len(LcList[b])
                    aa = Alist[b][offset + 2 * k:offset + 2 * k + 2,:]
                    N12[6 * b:6 * b + 6, 3 * i:3 * i + 3] += aa.T @ bb
                    U[6 * len(ImageList) + c:6 * len(ImageList) + c + 3, :] += bb.T @ ll
        c += 3
    sparse_matrices1 = [sc.dia_matrix(array) for array in N22BlockList]
    diagonal_matrix1 = sc.block_diag(sparse_matrices1)
    N22 = diagonal_matrix1.tobsr()

    L = np.concatenate(LList)
    Lvalue = np.linalg.norm(L)

    if lambdaa == np.inf:
        max1 = np.max(np.diag(N22.toarray()))
        max2 = np.min(np.diag(N11.toarray()))
        lambdaa = max(max1,max2)/1000

    elif Lvalue < dlLast and M:
        lambdaa = lambdaa/3
    else:
        lambdaa = lambdaa*2


    I22 = np.eye(N22.shape[0])*lambdaa
    I11 = np.eye(N11.shape[0])*lambdaa
    N22inverse = sc.bsr_matrix(InversBLockdiagonal3x3(N22.toarray()+I22))

    U1 = U[:6 * len(ImageList), 0]
    U2 = U[6*len(ImageList):, 0]


    return N11.toarray()+I11, N22inverse, U1, U2, N12, N22.toarray()+I22 , lambdaa, Lvalue

#######################################################################################################################################


# RandomInBlock   / firstImage /  Corner
ControlTemplate = "firstImage"
OriantationList, GroundTiepoints, ImageTiePointsList, ImageColntrolPointsList, GCPlist = syntheticImages(150, 230, 100,
                                                                                                         0.6, 3,
                                                                                                         ControlTemplate,
                                                                                                         5)

### Add Noises ###########################################################################################################
GTruthOrintation = []
GTruthTGP = []
for i in range(len(OriantationList)):
    GTruthOrintation.append(copy.copy(OriantationList[i]))
for i in range(len(GroundTiepoints)):
    GTruthTGP.append(copy.copy(GroundTiepoints[i]))

for img in OriantationList:
    noiseM = 5 * np.random.random(3)
    img["X0"] += noiseM[0]
    img["Y0"] += noiseM[1]
    img["Z0"] += noiseM[2]
    noiseS = 0.5*np.pi/180 * np.random.random(3)
    img["omega"] += noiseS[0]
    img["phi"] += noiseS[1]
    img["kappa"] += noiseS[2]

for I in range(len(ImageTiePointsList)):
    for P in ImageTiePointsList[I]:
        try:
            P.AddNoise(10 * 10 ** -3)
        except:
            pass

for I in range(len(ImageColntrolPointsList)):
    for P in ImageColntrolPointsList[I]:
        try:
            P.AddNoise(10 * 10 ** -3)
        except:
            pass


for CP in GroundTiepoints:
    CP.AddNoise(10)
######################################################################################################################################################################

SoulutionType = 10

####### Regular Sulotion #################################################################################################################################
Cam1=Camera(150, [0,0], [], [], [],230)
ImageList = []
for i in range(len(OriantationList)):
    ImageList.append(SingleImage(Cam1,i))
    ImageList[i].exteriorOrientationParameters = OriantationList[i]

GroundTiePoints = GroundPoint.listofPoints2Array(GroundTiepoints)
IDsGT = GroundTiePoints[:, 0]
GroundTiePoints = GroundTiePoints[:, 1:].astype(np.float)
Shapee=GroundTiePoints.shape
GroundTiePoints = GroundTiePoints.reshape(-1,)
dXo=[np.inf,np.Inf]

time1 = time()
######### Sparse Matrix Sulotion ##################################################################################################################

#the comment in green it is code for compering between the two way to calculate the N matrix.
#we convert it to comment of measure the time.


c=0

#Lambdaas = [0]
Lambdaas = [np.inf]
dxvalue = [np.inf]
dlvalue = [np.inf]
M = True
while True:
    if np.linalg.norm(dXo)>0.001:
        N11, N22inverse, U1, U2, N12 , N22, lambdaa, dll= ComputeNU(GroundTiePoints, IDsGT, GCPlist, Shapee, ImageTiePointsList, ImageColntrolPointsList, Lambdaas[-1], M, dlvalue[-1])
        Lambdaas.append(lambdaa)
        dlvalue.append(dll)
        try:
            N = N11 - N12 @ N22inverse @ N12.T
            U = U1 - N12 @ N22inverse @ U2
        except:
            lambdaa = Lambdaas[-1] * 2
            Lambdaas.append(lambdaa)
            dxvalue.append(np.inf)
            c+=1
            continue

        #####################################################################################
        if np.linalg.norm(dXo) == np.Inf:
            N4show1 = np.concatenate((N11, N12),axis=1)
            N4show2 = np.concatenate((N12.T, N22),axis=1)
            N4show = np.concatenate((N4show1,N4show2),axis=0)
            if np.linalg.norm(dXo) == np.Inf:
                plt.imshow((1 - np.ceil(abs(np.round(N4show,10)) / abs(np.round(N4show,10)).max())), cmap="gray")
                plt.show()
        #######################################################################################

        N = np.array(N)
        dXo = np.linalg.inv(N) @ U

        for j in range(len(ImageList)):
            ImageList[j].exteriorOrientationParameters["X0"] += dXo[j * 6]
            ImageList[j].exteriorOrientationParameters["Y0"] += dXo[j * 6 + 1]
            ImageList[j].exteriorOrientationParameters["Z0"] += dXo[j * 6 + 2]
            ImageList[j].exteriorOrientationParameters["omega"] += dXo[j * 6 + 3]
            ImageList[j].exteriorOrientationParameters["phi"] += dXo[j * 6 + 4]
            ImageList[j].exteriorOrientationParameters["kappa"] += dXo[j * 6 + 5]

        if np.linalg.norm(dXo) <= dxvalue[-1]:
            M = True
        else:
            M = False

        dxvalue.append(np.linalg.norm(dXo))
        c+=1

    #Last oiteration with lambda = 0
    else:
        N11, N22inverse, U1, U2, N12, N22, lambdaa, dll = ComputeNU(GroundTiePoints, IDsGT, GCPlist, Shapee,ImageTiePointsList, ImageColntrolPointsList,0, M, dlvalue[-1])
        Lambdaas.append(lambdaa)
        dlvalue.append(dll)

        N = np.array(N)
        dXo = np.linalg.inv(N) @ U

        for j in range(len(ImageList)):
            ImageList[j].exteriorOrientationParameters["X0"] += dXo[j * 6]
            ImageList[j].exteriorOrientationParameters["Y0"] += dXo[j * 6 + 1]
            ImageList[j].exteriorOrientationParameters["Z0"] += dXo[j * 6 + 2]
            ImageList[j].exteriorOrientationParameters["omega"] += dXo[j * 6 + 3]
            ImageList[j].exteriorOrientationParameters["phi"] += dXo[j * 6 + 4]
            ImageList[j].exteriorOrientationParameters["kappa"] += dXo[j * 6 + 5]

        if np.linalg.norm(dXo) <= dxvalue[-1]:
            M = True
        else:
            M = False

        dxvalue.append(np.linalg.norm(dXo))
        c += 1

        break

time2 = time()

K = np.linalg.cond(N4show)

diffOriantationList = []
for i in range(len(GTruthOrintation)):
    Result = OriantationList[i]
    GT = GTruthOrintation[i]
    diffOriantation = {}
    for key in Result:
        value1 = Result[key] # Get value from dict1
        value2 = GT.get(key, 0)  # Get value from dict2 if present, otherwise use 0
        diffOriantation[key] = abs(value1 - value2)  # Subtract values and store in result
        if key == "omega" or key == "kappa" or key == "phi":
            diffOriantation[key] = diffOriantation[key] * 206265
    diffOriantationList.append(diffOriantation)


print(f"the calculate time is {time2-time1} in second")

Diffarray = np.array([])
for i in range(len(diffOriantationList)):
    values = np.array(list(diffOriantationList[i].values()))
    Diffarray = np.concatenate((Diffarray,values))


Diffarray = Diffarray.reshape((len(diffOriantationList),6))
print(f"The average position error is {np.average(Diffarray[:,:3])} [m]")
print(f"The average Oriantation error is {np.average(Diffarray[:,3:])} ['']")
print(f"The condition number of N is {K*10**-6}")

Q = np.linalg.inv(N4show)
ru = np.eye(N4show.shape[0])

count = 0
maxx=-np.inf
minn = np.inf
summ = 0
maxindex = []

for i in range(N4show.shape[0]):
    for j in range(i+1,N4show.shape[0]):
        ru[j,i] = Q[j,i] / np.sqrt(Q[i,i]*Q[j,j])
        count+=1
        summ += abs(ru[j,i])
        if abs(ru[j,i]) > maxx:
            maxx = abs(ru[j,i])
        if abs(ru[j,i]) < minn:
            minn = abs(ru[j,i])
        maxindex.append([j,i,abs(ru[j,i])])

maxindex = np.array(maxindex)
sorted_indices = np.argsort(maxindex[:, 2])
maxindex = maxindex[sorted_indices,:]

print(f"the average absolute corelation is {summ / count}, the maximum is {maxx} and the minimum is {minn}")

MM.PrintMatrix(np.array([summ / count, maxx, minn]))
ru[np.abs(ru)<0.5] = 0
plt.imshow((np.abs(ru)),cmap="gray")
plt.show()


fig3 = plt.figure()
ax31 = fig3.add_subplot(131)
ax32 = fig3.add_subplot(132)
ax33 = fig3.add_subplot(133)

ax31.plot(Lambdaas)
ax32.plot(dxvalue)
ax33.plot(dlvalue)

ax31.set_xlabel("Iteration Number")
ax32.set_xlabel("Iteration Number")
ax33.set_xlabel("Iteration Number")

ax31.set_ylabel("Lambda")
ax32.set_ylabel("dX")
ax33.set_ylabel("dV")


plt.show()


