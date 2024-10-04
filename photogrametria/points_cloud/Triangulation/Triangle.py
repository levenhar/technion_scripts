import random

import matplotlib.pyplot as plt
import numpy as np


class Triangle():
    def __init__(self, ID, ListOfVertex, Neighbors, AllVertex):
        self.__id = ID
        self.ifFlip = False
        if not self.ClockwiseORNot(ListOfVertex, AllVertex):
            temp =ListOfVertex[0]
            temp1 = ListOfVertex[-1]
            ListOfVertex[-1] = temp
            ListOfVertex[0] = temp1
            self.ifFlip = True

        self.__ListOfVertex = ListOfVertex
        self.__Neighbors = Neighbors

    @property
    def ID(self):
        return self.__id

    @property
    def ListOfVertex(self):
        return self.__ListOfVertex

    @property
    def Neighbors(self):
        return self.__Neighbors

    @Neighbors.setter
    def Neighbors(self, value):
        if self.ifFlip:
            temp = value[0]
            temp1 = value[-1]
            value[0] = temp1
            value[-1] = temp
        self.__Neighbors = value

    def Centroid(self, Allvertex):
        P1 = Allvertex[self.ListOfVertex[0], :].reshape((1, 3))
        P2 = Allvertex[self.ListOfVertex[1], :].reshape((1, 3))
        P3 = Allvertex[self.ListOfVertex[2], :].reshape((1, 3))
        M = np.concatenate((P1, P2, P3))

        C = np.average(M,0)
        return C

    def generate_point_inside_triangle(self, Allvertex):
        # Generate a random point inside the triangle defined by vertex_A, vertex_B, and vertex_C
        vertex_A = Allvertex[self.ListOfVertex[0], :]
        vertex_B = Allvertex[self.ListOfVertex[1], :]
        vertex_C = Allvertex[self.ListOfVertex[2], :]
        # Generate random barycentric coordinates
        u = np.random.random()
        v = np.random.random()

        # Ensure the point lies within the triangle
        while  u + v > 0.66 or u + v <0.33:
            u = np.random.random()
            v = np.random.random()

        # Calculate the coordinates of the point inside the triangle
        w = 1 - u - v
        point_x = u * vertex_A[0] + v * vertex_B[0] + w * vertex_C[0]
        point_y = u * vertex_A[1] + v * vertex_B[1] + w * vertex_C[1]
        #plt.scatter(point_x,point_y)
        return np.array([point_x, point_y])

    def ClockwiseORNot(self, ListOfVertex, Allvertex):
        P1 = Allvertex[ListOfVertex[0],:2].reshape((1,2))
        P2 = Allvertex[ListOfVertex[1],:2].reshape((1,2))
        P3 = Allvertex[ListOfVertex[2],:2].reshape((1,2))
        M = np.concatenate((P1,P2,P3))
        M = np.concatenate((M, np.ones((3, 1))), axis=1)
        if np.linalg.det(M) >= 0:
            return True
        else:
            return False

    def plot(self,withtext, color,Allvertex, ax=None):
        ids = self.ListOfVertex + [self.ListOfVertex[0]]
        Vertex2plot = Allvertex[ids,:]
        if ax == None:
            plt.plot(Vertex2plot[:, 0], Vertex2plot[:, 1],c=color)
            if withtext:
                plt.text(np.average(Allvertex[self.ListOfVertex,0]),np.average(Allvertex[self.ListOfVertex,1]),f"{self.ID}")
        else:
            ax.plot(Vertex2plot[:, 0], Vertex2plot[:, 1], c=color)
            if withtext:
                ax.text(np.average(Allvertex[self.ListOfVertex,0]),np.average(Allvertex[self.ListOfVertex,1]),f"{self.ID}")


    def Point_inTriangel(self, Point, Allvertex):
        P1 = Allvertex[self.ListOfVertex[0], :].reshape((1, 3))
        P2 = Allvertex[self.ListOfVertex[1], :].reshape((1, 3))
        P3 = Allvertex[self.ListOfVertex[2], :].reshape((1, 3))
        POINTS = np.concatenate((P1,P2,P3))
        vecs = POINTS - Point.reshape(1, 3)
        vecs = vecs[:,:2]
        Norms = np.linalg.norm(vecs,axis=1)
        summ = 0
        for i in range(3):
            mone = vecs[i].reshape((1,2))@vecs[i-1].reshape((2,1))
            summ += np.arccos(mone/(Norms[i]*Norms[i-1]))
        if round(summ[0,0],5)==round(2*np.pi,5):
            return True
        else:
            return False

    def InBlockTriangle(self, CheckedPoint, Allvertex):
        P1 = Allvertex[self.ListOfVertex[0], :2].reshape((1, 2))
        P2 = Allvertex[self.ListOfVertex[1], :2].reshape((1, 2))
        P3 = Allvertex[self.ListOfVertex[2], :2].reshape((1, 2))
        TriPoints = np.concatenate((P1, P2, P3))
        #TriPoints = self.ListOfVertex[:,:2]
        TriPoints = np.concatenate((TriPoints, CheckedPoint[:,:2]))
        TriPoints = np.concatenate((TriPoints, (TriPoints[:,0]**2+TriPoints[:,1]**2).reshape((4,1)),np.ones((4,1))),axis=1)
        D = np.linalg.det(TriPoints)
        if D >= 0:
            return True
        else:
            return False


    def __eq__(self, other):
        if type(other) == str or type(other) == np.str_:
            if self.ID == other:
                return True
            else:
                return False
        elif type(other) == Triangle:
            if self.ID == other.ID:
                return True
            else:
                return False
        elif other == None:
            return False
        else:
            raise Exception("the ID need to be string")

    def __repr__(self):
        return f"T{self.ID}"
