import pandas as pd
import numpy as np



def readsp3(SP3NameFile):
    df = pd.read_fwf(SP3NameFile, header=None)
    ar = df.to_numpy()[:,0]
    ar = np.delete(ar,-1)
    ar = ar.reshape(len(ar),1)
    Information=[]
    for i in range(len(ar)):
        if ar[i,0][0]=="*":
            c=-1
            q=0
            CoordinatesAll = np.zeros((31,4,int((len(ar)-i)/32)))
            for j in range(i,len(ar),32):
                Coordinates = np.array(["0","0","0","0","0"],str).reshape(1,5)
                for p in range(32):
                    if ar[j+p, 0][0] == "*":
                        info=ar[j+p,0].split(" ")
                        for k in info:
                            if k == "":
                                info.remove("")
                        Information.append(info)
                        c+=1
                    else:
                        satellite=ar[j+p,0].split(" ")
                        satellite=np.array(satellite)
                        revindex=[]
                        for k in range(len(satellite)):
                            if satellite[k] == "":
                                revindex.append(k)
                                #satellite.remove("")
                        satellite=np.delete(satellite,revindex).reshape(1,5)
                        Coordinates=np.concatenate((Coordinates,satellite),axis=0)
                        q+=1
                Coordinates=np.delete(Coordinates,0,0)
                CoordinatesAll[:,:,c]=Coordinates[:,1:]
            Satell=Coordinates[:,0]
            return CoordinatesAll, Information, Satell




def readNfile(NfailName):
    df = pd.read_fwf(NfailName, header=None)
    ar=df.to_numpy()

    for i in range(len(ar)):
        while ar[i] == "END OF HEADER":
            Nfile = np.zeros((8,4,int((len(ar)-i-1)/8)))
            Dates = []
            #np.zeros((int((len(ar)-i-1)/8),1),str)
            c=0
            for j in range(i+1,len(ar),8):
                Mat=np.zeros((8,4))
                for k in range(8):
                    row=np.zeros((1,4))
                    row[0,3] = float(ar[j+k, 0][-19:-4]) * 10 ** int(ar[j+k, 0][-3:])
                    row[0,2] = float(ar[j+k, 0][-38:-23]) * 10 ** int(ar[j+k, 0][-22:-19])
                    row[0,1] = float(ar[j+k, 0][-57:-42]) * 10 ** int(ar[j+k, 0][-41:-38])
                    if k==0:
                        Dates.append((ar[j + k, 0][:-57]))
                    else:
                        row[0,0] = float(ar[j+k, 0][:-61]) * 10 ** int(ar[j+k, 0][-60:-57])
                    Mat[k,:]=row
                Nfile[:,:,c]=Mat
                c+=1
            return Nfile, np.array(Dates).reshape((len(Dates),1))


def readOfile(OfailName):
    df = pd.read_fwf(OfailName, header=None)
    ar = df.to_numpy()
    aprx=ar[10,0].split(" ")
    removList = []
    for p in range(len(aprx)):
        if aprx[p] == "":
            removList.append(p)
    aprx = np.array(aprx)
    aprx = np.delete(aprx, removList)
    for i in range(len(ar)):
        if ar[i] == "END OF HEADER":
            info=np.array([0,0,0,0,0,0,0,0]).reshape((1,8))
            satelliteNum=np.array([0,0,0,0,0,0,0,0,0,0,0,0],str).reshape((1,12))
            Observations=np.zeros((12,4,1))
            c=0
            y=0
            #len(ar)
            for j in range(i+1,len(ar)):
                if j+y>len(ar)-1:
                    break
                #header of epoch
                Inf = ar[j+y, 0][:31].split(" ")
                removList = []
                for p in range(len(Inf)):
                    if Inf[p] == "":
                        removList.append(p)
                Inf = np.array(Inf)
                try:
                    Inf = np.delete(Inf, removList).reshape((1, 8))
                except:
                    pass
                info = np.concatenate((info, Inf), axis=0)
                S = ar[j+y, 0][32:].split("G")
                S=np.array(S)
                if len(S)<12:
                    l=12-len(S)
                    S=np.concatenate((S,np.zeros(l)))
                satelliteNum=np.concatenate((satelliteNum,S.reshape((1,12))),axis=0)
                ########################
                mat=np.zeros((12,4))
                k=0
                #y+=1
                try:
                    mm=ar[j + y + 1, 0][:2]
                except:
                    mm="18"
                while mm != "18":
                    try:
                        row=ar[j+y+1,0].split(" ")
                    except:
                        pass
                    removList=[]
                    for p in range(len(row)):
                        if row[p]=="" or len(row[p])==1:
                            removList.append(p)
                    row=np.array(row)
                    row=np.delete(row,removList)
                    if len(row)<4:
                        q=4-len(row)
                        qq=np.zeros((q))
                        row=np.concatenate((row,qq),axis=0).reshape((1,4))
                    else:
                        row=row.reshape((1, 4))
                    try:
                        mat[k,:]=row
                    except:
                        pass
                    k+=1
                    y+=1
                    try:
                        mm = ar[j + y + 1, 0][:2]
                    except:
                        mm = "18"
                if 0<k:
                    Observations=np.concatenate((Observations,mat.reshape(12,4,1)),axis=2)
                    c+=1
            return Observations , np.array(satelliteNum), info , aprx


