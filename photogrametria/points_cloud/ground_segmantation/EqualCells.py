from PointsCloud import PointsCloud
import numpy as np


class EqualCells:
    def __init__(self, Points, n0):
        '''

        :param Points:
        :type Points:PointsCloud
        '''
        self.n0=n0 #average number of point in a cell
        self.cloud=Points #save the points in Points Cloud


        #save the minimum and maxsimum value in each axis
        self.Xmin = min(self.cloud["x"])
        self.Ymin = min(self.cloud["y"])
        self.Zmin = min(self.cloud["z"])
        self.Xmax = max(self.cloud["x"])
        self.Ymax = max(self.cloud["y"])
        self.Zmax = max(self.cloud["z"])

        Ly = self.Ymax - self.Ymin
        Lx = self.Xmax - self.Xmin
        Lz = self.Zmax - self.Zmin
        P = len(self.cloud.pointsC) / self.n0  #number of cells
        cellArea = (Lx * Ly * Lz) / P #valuem of each cell
        self.l = (cellArea) ** 0.333333

        #number of cells in each axis
        Numx = int(np.ceil(Lx / self.l))
        Numy = int(np.ceil(Ly / self.l))
        Numz = int(np.ceil(Lz / self.l))
        ##############################################

        Xarr = self.cloud["x"]
        Yarr = self.cloud["y"]
        Zarr = self.cloud["z"]

        self.DBMatrix = [[[[] for k in range(Numz)] for j in range(Numx)] for i in range(Numy)] # creating an empty matrix.

        for i in range(len(self.cloud.pointsC)):  # loop for place the indexes of the coordinate in the relevant cell.
            X = Xarr[i]
            Y = Yarr[i]
            Z = Zarr[i]
            ny, nx, nz = self.XYZ2IJK(int(Y), int(X), int(Z))
            self.DBMatrix[ny][nx][nz].append(i)

    def XYZ2IJK(self, Y, X, Z):
        '''
        The function get N,E coordinate and return the i,j indexes in the DB matrix
        '''
        i = int(np.floor((Y - self.Ymin) / self.l))
        if i < 0:
            i = 0
        if i > len(self.DBMatrix) - 1:
            i = len(self.DBMatrix) - 1
        j = int(np.floor((X - self.Xmin) / self.l))
        if j < 0:
            j = 0
        if j > len(self.DBMatrix[0]) - 1:
            j = len(self.DBMatrix[0]) - 1
        k = int(np.floor((Z - self.Zmin) / self.l))
        if k < 0:
            k = 0
        if k > len(self.DBMatrix[0][0]) - 1:
            k = len(self.DBMatrix[0][0]) - 1
        return i, j , k