import numpy as np
from Triangle import Triangle
import time
import vtk
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union



class Triangulation():
    def __init__(self,Dataarray, idoffset = 0):

        self.maxX = np.max(Dataarray[:,0])
        self.maxY = np.max(Dataarray[:,1])
        self.minX = np.min(Dataarray[:,0])
        self.minY = np.min(Dataarray[:,1])

        Dx = self.maxX - self.minX
        Dy = self.maxY - self.minY
        M = max(Dx, Dy)
        XcYc = np.average(Dataarray, 0)[:2]

        triangle_points = np.array([[XcYc[0] + 3 * M, XcYc[1], 0], [XcYc[0], XcYc[1] + 3 * M, 0], [XcYc[0] - 3 * M, XcYc[1] - 3 * M, 0]])
        Dataarray = np.concatenate((Dataarray,triangle_points))

        self.Vertex = Dataarray
        T1 = Triangle("0", [-3,-2,-1], np.array([[None, None, None]]).T,self.Vertex)
        self.Triangles = [T1]
        StockTriangle = []

        time1 = time.time()
        c = 1
        for i in range(len(Dataarray)-3):
            if all(Dataarray[i,:] == np.zeros(3)):
                continue
            j = self.PointInTriangle(Dataarray[i, :])
            P = Dataarray[i, :].reshape((1, 3))
            T1 = Triangle(f"{idoffset+c}", self[j].ListOfVertex[:2] + [i], np.array([[None, None, None]]).T, self.Vertex)
            T2 = Triangle(f"{idoffset+c + 1}", self[j].ListOfVertex[1:] + [i], np.array([[None, None, None]]).T, self.Vertex)
            T3 = Triangle(f"{idoffset+c + 2}", [self[j].ListOfVertex[0], self[j].ListOfVertex[2]] + [i], np.array([[None, None, None]]).T, self.Vertex)


            T1.Neighbors = np.array([T2.ID, T3.ID, self[j].Neighbors[2]], dtype=object)
            T2.Neighbors = np.array([T3.ID, T1.ID, self[j].Neighbors[0]], dtype=object)
            T3.Neighbors = np.array([T2.ID, T1.ID, self[j].Neighbors[1]], dtype=object)

            for m in range(len(self[j].Neighbors)):
                if self[j].Neighbors[m] != None:
                    PointsN = self.Vertex[self[self[j].Neighbors[m]].ListOfVertex,:]
                    PointsSelf = self.Vertex[self[j].ListOfVertex]
                    for q in range(len(PointsN)):
                        if not np.any(np.all(PointsSelf == PointsN[q, :], axis=1)):
                            if m == 0:
                                self[self[j].Neighbors[m]].Neighbors[q] = T2.ID
                            elif m == 1:
                                self[self[j].Neighbors[m]].Neighbors[q] = T3.ID
                            else:
                                self[self[j].Neighbors[m]].Neighbors[q] = T1.ID
                            break

            self = self - j
            self.Triangles.extend([T1, T2, T3])
            StockTriangle.extend([T1, T2, T3])

            c += 3

            while len(StockTriangle) > 0:
                CheckT = StockTriangle.pop()
                for q in range(3):
                    if all(self.Vertex[CheckT.ListOfVertex[q],:] == P.reshape(-1, )) and CheckT.Neighbors[q] != None:
                        TriPoints = self.Vertex[CheckT.ListOfVertex, :2]
                        for n in range(3):
                            if not np.any(np.all(self.Vertex[self[CheckT.Neighbors[q]].ListOfVertex[n], :].reshape(
                                    (1, 3)) == self.Vertex[CheckT.ListOfVertex,:], axis=1)):
                                CheckedPoint = self.Vertex[self[CheckT.Neighbors[q]].ListOfVertex[n], :].reshape((1, 3))
                                break
                        OtherTriangle = self[CheckT.Neighbors[q]]

                        if CheckT.InBlockTriangle(CheckedPoint, self.Vertex):

                            NewPoints = [i, self[CheckT.Neighbors[q]].ListOfVertex[n]]

                            T1 = Triangle(f"{idoffset+c}",
                                          NewPoints + [CheckT.ListOfVertex[q - 1]],
                                          np.array([[None, None, None]]).T, self.Vertex)
                            T2 = Triangle(f"{idoffset+c + 1}",
                                         NewPoints + [CheckT.ListOfVertex[q - 2]],
                                          np.array([[None, None, None]]).T, self.Vertex)

                            T1.Neighbors = np.array([OtherTriangle.Neighbors[n - 1], CheckT.Neighbors[q - 2], T2.ID],
                                                    dtype=object)
                            T2.Neighbors = np.array([OtherTriangle.Neighbors[n - 2], CheckT.Neighbors[q - 1], T1.ID],
                                                    dtype=object)

                            self.Triangles.extend([T1, T2])

                            #########################################################################################################
                            ########  Neighbors corection ##################################################################################
                            NeihgborID = CheckT.Neighbors[q - 1]
                            if NeihgborID != None:
                                for m in range(len(CheckT.ListOfVertex)):
                                    if self[NeihgborID] != None:
                                        if not np.any(np.all(self.Vertex[self[NeihgborID].ListOfVertex[m], :].reshape(
                                                (1, 3)) == self.Vertex[CheckT.ListOfVertex,:], axis=1)):
                                            self[NeihgborID].Neighbors[m] = T2.ID

                            NeihgborID = CheckT.Neighbors[q - 2]
                            if NeihgborID != None:
                                for m in range(len(CheckT.ListOfVertex)):
                                    if self[NeihgborID] != None:
                                        if not np.any(np.all(self.Vertex[self[NeihgborID].ListOfVertex[m], :].reshape(
                                                (1, 3)) == self.Vertex[CheckT.ListOfVertex,:], axis=1)):
                                            self[NeihgborID].Neighbors[m] = T1.ID

                            NeihgborID = OtherTriangle.Neighbors[n - 1]
                            if NeihgborID != None:
                                for m in range(len(OtherTriangle.ListOfVertex)):
                                    if self[NeihgborID] != None:
                                        if not np.any(np.all(self.Vertex[self[NeihgborID].ListOfVertex[m], :].reshape(
                                                (1, 3)) == self.Vertex[OtherTriangle.ListOfVertex,:], axis=1)):
                                            self[NeihgborID].Neighbors[m] = T1.ID

                            NeihgborID = OtherTriangle.Neighbors[n - 2]
                            if NeihgborID != None:
                                for m in range(len(OtherTriangle.ListOfVertex)):
                                    if self[NeihgborID] != None:
                                        if not np.any(np.all(self.Vertex[self[NeihgborID].ListOfVertex[m], :].reshape(
                                                (1, 3)) == self.Vertex[OtherTriangle.ListOfVertex,:], axis=1)):
                                            self[NeihgborID].Neighbors[m] = np.str_(T2.ID[:])
                            #################################################################################################

                            self = self - CheckT
                            self = self - OtherTriangle

                            StockTriangle.append(T1)
                            StockTriangle.append(T2)

                            c += 2
                            break

        time2 = time.time()
        DellList = []
        for T in self.Triangles:
            m = []
            for p in range(len(T.ListOfVertex)):
                if T.ListOfVertex[p] < 0:
                    m.append(p)
            if len(m) == 0:
                continue
            elif len(m) == 1:
                Neihbor = T.Neighbors[m[0]]
                for nn in range(len(self[Neihbor].ListOfVertex)):
                    if not np.any(np.all(self.Vertex[self[Neihbor].ListOfVertex[nn], :].reshape((1, 3)) == self.Vertex[T.ListOfVertex,:],axis=1)):
                        self[Neihbor].Neighbors[nn] = None
            DellList.append(T.ID)
            # self = self - T

        self = self - DellList

        time3 = time.time()
        print(f"Tirangolation time - {(time2 - time1) / 60} [minutes]")
        print(f"Deleting time - {(time3 - time2) / 60} [minutes]")

        self.Vertex = np.delete(self.Vertex,[-1,-2,-3],axis = 0)

    def __len__(self):
        return len(self.Triangles)

    def __getitem__(self, item):
        if type(item) == str or type(item) == np.str_:
            for i in range(len(self.Triangles)):
                if self.Triangles[i].ID == item:
                    return self.Triangles[i]
            raise Exception("the input Triangle does not exist")
        elif type(item) == int and item < len(self.Triangles):
            return self.Triangles[item]
        else:
            raise Exception("the input is incorrect")


    def __sub__(self, other):
        if type(other) == Triangle or type(other) == str:
            for i in range(len(self.Triangles)):
                if self.Triangles[i] == other:
                    del self.Triangles[i]
                    break
        elif type(other) == list:
            for TT in other:
                for i in range(len(self.Triangles)):
                    if self.Triangles[i] == TT:
                        del self.Triangles[i]
                        break
        return self

    def get_angle(self,x, y):
        # Calculate the angle in radians
        Center = np.average(self.Vertex,axis=0)
        return np.arctan2(y-Center[1], x-Center[0])

    def order_rows_clockwise(self,arr):
        # Get the coordinates of each row
        x_coords = arr[:, 0]
        y_coords = arr[:, 1]

        # Calculate the angles for each row
        angles = self.get_angle(x_coords, y_coords)

        # Sort the rows based on the angles in clockwise order
        sorted_indices = np.argsort(angles)
        sorted_arr = arr[sorted_indices]

        return sorted_arr


    def ConvexHull(self):
        CH = []
        for T in self.Triangles:
            MM=[]
            for p in range(len(T.Neighbors)):
                if T.Neighbors[p] == None:
                    MM.append(p)
            if len(MM) == 0:
                continue
            elif len(MM) == 2:
                CH.extend(T.ListOfVertex)
            else:
                indexx = [0,1,2]
                for m in MM:
                    del indexx[m]
                CH.extend([T.ListOfVertex[indexx[0]],T.ListOfVertex[indexx[1]]])
        CH = np.unique(np.array(CH), axis=0)
        CH = self.Vertex[CH,:2]
        CH = self.order_rows_clockwise(CH)
        CH = np.concatenate((CH, CH[0,:].reshape((1,2))))
        return CH



    def PointHight(self,Point):
        j = self.PointInTriangle(Point)
        if any(self[j].Neighbors == None):
            w=[]
            z=[]
            for i in range(len(self[j].ListOfVertex)):
                z.append(self.Vertex[self[j].ListOfVertex[i],2])
                d = (self.Vertex[self[j].ListOfVertex[i],:2].reshape((2,1))-Point[:2,:]).T@(self.Vertex[self[j].ListOfVertex[i],:2].reshape((2,1))-Point[:2,:])
                w.append(1/(0.00001 + np.sqrt(d[0,0])))
            w = np.array(w)
            z = np.array(z)
            Z = z.T@w/np.sum(w)
            return Z, self[j]
        else:
            POINTS = self.Vertex[self[j].ListOfVertex,:]
            for n in self[j].Neighbors:
                POINTS = np.concatenate((POINTS, self.Vertex[self[n].ListOfVertex,:]))
            POINTS = np.unique(POINTS, axis=0)
            b = POINTS[:,2]
            A = np.concatenate((np.ones((6,1)),POINTS[:,:2]),axis=1)
            A = np.concatenate((A, POINTS[:,:2]**2, (POINTS[:,0]*POINTS[:,1]).reshape((6,1))),axis=1)
            X = np.linalg.solve(A,b).reshape(6,1)
            M = np.array([[1, Point[0,0], Point[1,0], Point[0,0]**2, Point[1,0]**2, Point[0,0]*Point[1,0]]])
            Z = M@X
            return Z[0,0], self[j]

    def HightSection(self,P1,P2):
        H1, T1 = self.PointHight(P1)
        H2, T2 = self.PointHight(P2)
        section = np.array([[0,H1]])
        Points = np.array([[P1[0,0],P1[1,0],H1]])
        if T1 == T2:
            section = np.concatenate((section, np.array([[1, H2]])))
            Points = np.concatenate((Points, np.array([[P2[0, 0], P2[1, 0], H2]])))
            return section, Points
        c = 0
        while T1 != T2:
            t = 1
            for i in range(len(T1.ListOfVertex)):
                P3 = self.Vertex[T1.ListOfVertex[i],:].reshape(-1,)
                P4 = self.Vertex[T1.ListOfVertex[i-1], :].reshape(-1,)
                ti = ((P1[0]-P3[0])*(P4[1]-P3[1])-(P4[0]-P3[0])*(P1[1]-P3[1]))/((P4[1]-P3[1])*(P1[0]-P2[0])-(P4[0]-P3[0])*(P1[1]-P2[1]))
                if round(ti[0],12)>round(c,12) and ti[0]<t:
                    t = ti[0]
                    inx = i
            c=t
            P3 = self.Vertex[T1.ListOfVertex[inx], :].reshape(-1, )
            P4 = self.Vertex[T1.ListOfVertex[inx - 1], :].reshape(-1, )
            t2 = ((P1[0]-P3[0])-(P1[0]-P2[0])*t)/(P4[0]-P3[0])
            z = P3[2]*(1-t2)+P4[2]*t2
            x = P1[0]*(1-t)+P2[0]*t
            y = P1[1] * (1 - t) + P2[1] * t
            section = np.concatenate((section,np.array([[t,z[0]]])))
            Points = np.concatenate((Points,np.array([[x[0],y[0],z[0]]])))
            T1 = self[T1.Neighbors[inx-2]]
        section = np.concatenate((section, np.array([[1, H2]])))
        Points = np.concatenate((Points, np.array([[P2[0,0],P2[1,0],H2]])))
        return section, Points


    def PointInTriangle(self,Point):
        if not self.minX <= Point[0] <= self.maxX and self.minY <= Point[1] <= self.maxY:
            raise Exception("the point is outside of all the triangles")
        T1 = self.Triangles[0]
        P2 = Point.reshape(-1, )
        while True:
            if T1.Point_inTriangel(Point, self.Vertex):
                return T1.ID
            P1 = T1.Centroid(self.Vertex).reshape(-1,)
            t = 1
            for i in range(len(T1.ListOfVertex)):
                P3 = self.Vertex[T1.ListOfVertex[i]].reshape(-1,)
                P4 = self.Vertex[T1.ListOfVertex[i-1]].reshape(-1,)
                ti = ((P1[0]-P3[0])*(P4[1]-P3[1])-(P4[0]-P3[0])*(P1[1]-P3[1]))/((P4[1]-P3[1])*(P1[0]-P2[0])-(P4[0]-P3[0])*(P1[1]-P2[1]))
                if ti>0 and ti<t:
                    t = ti
                    Ninx = i

            if T1.Neighbors[Ninx-2] == None:
                raise Exception("Error")

            T1=self[T1.Neighbors[Ninx-2]]

    def ConvertToarray(self):
        TrianglesVertecs = []
        for T in self.Triangles:
            V = T.ListOfVertex
            TrianglesVertecs.append(V)
        TrianglesVertecs = np.array(TrianglesVertecs)
        return TrianglesVertecs


    def AddConstraints(self,constraint):
        ### find the triangle that we need to erase ########################################################
        V = self.Vertex[constraint[1]] - self.Vertex[constraint[0]]
        P1 = (self.Vertex[constraint[0]] + V*0.01).reshape(-1,)
        P2 = (self.Vertex[constraint[1]] - V*0.01).reshape(-1,)
        T1 = self.PointInTriangle(P1)
        T2 = self.PointInTriangle(P2)
        if T1 == T2:
            return self
        Triangle2Remove = [T1,T2]
        c = 0
        while T1 != T2:
            t = 1
            for i in range(len(self[T1].ListOfVertex)):
                P3 = self.Vertex[self[T1].ListOfVertex[i], :].reshape(-1, )
                P4 = self.Vertex[self[T1].ListOfVertex[i - 1], :].reshape(-1, )
                ti = ((P1[0] - P3[0]) * (P4[1] - P3[1]) - (P4[0] - P3[0]) * (P1[1] - P3[1])) / (
                            (P4[1] - P3[1]) * (P1[0] - P2[0]) - (P4[0] - P3[0]) * (P1[1] - P2[1]))
                if round(ti, 12) > round(c, 12) and ti < t:
                    t = ti
                    inx = i
            c = t
            T1 = self[T1].Neighbors[inx - 2]
            Triangle2Remove.append(T1)
        ###############################################################################################################################################

        #### create a list of the vertex the we need for the new triangles #######################################################
        Vertexx = []
        for t in Triangle2Remove:
            Vertexx.extend(self[t].ListOfVertex)
        Vertexx = np.unique(np.array(Vertexx), axis=0)
        ###############################################################################################################################################

        #### split the vertex into 2 groups, one right to the Constraint and the second in the left ##################################
        Right = []
        Left = []
        for i in Vertexx:
            Vi = (self.Vertex[i,:2]-self.Vertex[constraint[0],:2])
            VV = V[:2]
            if np.cross(Vi,VV)>0:
                Right.append(i)
            elif np.cross(Vi,VV)<0:
                Left.append(i)
            else:
                Right.append(i)
                Left.append(i)

        RightData = np.zeros((self.Vertex.shape))
        LeftData = np.zeros((self.Vertex.shape))
        RightData[Right, :] = self.Vertex[Right,:]
        LeftData[Left, :] = self.Vertex[Left, :]
        ###############################################################################################################################################

        ### Create new Triangulation for each side of the Constraint ########################################################
        c = int(self.Triangles[-1].ID)+1
        RightTriangulation = Triangulation(RightData,c)
        c += int(RightTriangulation.Triangles[-1].ID)+1
        LeftTriangulation = Triangulation(LeftData, c)

        #############################################################################################################################
        ChangeArea = []
        for T in Triangle2Remove:
            ChangeArea.append(Polygon(self.Vertex[self[T].ListOfVertex,:2]))
        ChangeArea = unary_union(ChangeArea)

        ####### Add Triangle near the constraint if needed ######################################################################################################################
        # for Right side
        for T in RightTriangulation.Triangles:
            if self.Vertex[constraint[1], :] in RightTriangulation.Vertex[T.ListOfVertex, :] and self.Vertex[constraint[0],:] in RightTriangulation.Vertex[T.ListOfVertex,:]:
                break
        else:
            mark1 = []
            mark0 = []
            for TTT in RightTriangulation.Triangles:
                if self.Vertex[constraint[1], :] in RightTriangulation.Vertex[TTT.ListOfVertex, :]:
                    mark1.append(TTT.ID)
                elif self.Vertex[constraint[0], :] in RightTriangulation.Vertex[TTT.ListOfVertex, :]:
                    mark0.append(TTT.ID)
            for m1 in mark1:
                for m0 in mark0:
                    for i in range(3):
                        if RightTriangulation[m1].ListOfVertex[i] in RightTriangulation[m0].ListOfVertex:
                            Tri1 = m1
                            Tri0 = m0
                            vertex2 = RightTriangulation[m1].ListOfVertex[i]
                            break

            T1 = Triangle(str(int(LeftTriangulation.Triangles[-1].ID) + 1), constraint + [vertex2],np.array([Tri1, Tri0, None]), RightData)

            for q in range(len(RightTriangulation[Tri0].ListOfVertex)):
                if not np.any(np.all(self.Vertex[RightTriangulation[Tri0].ListOfVertex[q], :].reshape((1, 3)) == self.Vertex[T1.ListOfVertex, :],axis=1)):
                    RightTriangulation[Tri0].Neighbors[q] = T1.ID

            for q in range(len(RightTriangulation[Tri1].ListOfVertex)):
                if not np.any(np.all(self.Vertex[RightTriangulation[Tri1].ListOfVertex[q], :].reshape((1, 3)) == self.Vertex[T1.ListOfVertex, :],axis=1)):
                    RightTriangulation[Tri1].Neighbors[q] = T1.ID
            RightTriangulation.Triangles.append(T1)

        # for Left side
        for T in LeftTriangulation.Triangles:
            if self.Vertex[constraint[1], :] in LeftTriangulation.Vertex[T.ListOfVertex, :] and self.Vertex[constraint[0],:] in LeftTriangulation.Vertex[T.ListOfVertex,:]:
                break
        else:
            mark1=[]
            mark0=[]
            for TTT in LeftTriangulation.Triangles:
                if self.Vertex[constraint[1], :] in LeftTriangulation.Vertex[TTT.ListOfVertex, :]:
                    mark1.append(TTT.ID)
                elif self.Vertex[constraint[0], :] in LeftTriangulation.Vertex[TTT.ListOfVertex, :]:
                    mark0.append(TTT.ID)
            for m1 in mark1:
                for m0 in mark0:
                    for i in range(3):
                        if LeftTriangulation[m1].ListOfVertex[i] in LeftTriangulation[m0].ListOfVertex:
                            Tri1 = m1
                            Tri0 = m0
                            vertex2 = LeftTriangulation[m1].ListOfVertex[i]
                            break

            T1 = Triangle(str(int(LeftTriangulation.Triangles[-1].ID)+1),constraint+[vertex2],np.array([Tri1,Tri0,None]),LeftData)
            for q in range(len(LeftTriangulation[Tri0].ListOfVertex)):
                if not np.any(np.all(self.Vertex[LeftTriangulation[Tri0].ListOfVertex[q], :].reshape((1, 3)) == self.Vertex[T1.ListOfVertex, :], axis=1)):
                    LeftTriangulation[Tri0].Neighbors[q] = T1.ID
            for q in range(len(LeftTriangulation[Tri1].ListOfVertex)):
                if not np.any(np.all(self.Vertex[LeftTriangulation[Tri1].ListOfVertex[q], :].reshape((1, 3)) == self.Vertex[T1.ListOfVertex, :], axis=1)):
                    LeftTriangulation[Tri1].Neighbors[q] = T1.ID
            LeftTriangulation.Triangles.append(T1)


        ##### erase triangle that dont fit in the area of the deleted triangles #########################################
        Removelist = []
        TriangleList = np.copy(RightTriangulation.Triangles)
        while len(TriangleList) != 0:
            T = TriangleList[0]
            TriangleList = np.delete(TriangleList, 0)
            C = Polygon(self.Vertex[T.ListOfVertex,:2])
            if not ChangeArea.contains(C) and any(T.Neighbors == None):
                Removelist.append(T)
                for NeihgborID in T.Neighbors:
                    if NeihgborID != None:
                        for m in range(len(RightTriangulation[NeihgborID].ListOfVertex)):
                            if not RightTriangulation[NeihgborID].ListOfVertex[m] in T.ListOfVertex:
                                RightTriangulation[NeihgborID].Neighbors[m] = None
                        TriangleList = np.concatenate((TriangleList, np.array([RightTriangulation[NeihgborID]])))
        RightTriangulation = RightTriangulation - Removelist


        Removelist = []
        TriangleList = np.copy(LeftTriangulation.Triangles)
        while len(TriangleList) != 0:
            T = TriangleList[0]
            TriangleList = np.delete(TriangleList, 0)
            C = Polygon(self.Vertex[T.ListOfVertex,:2])
            #C = Point(T.generate_point_inside_triangle(LeftData))
            if not ChangeArea.contains(C) and any(T.Neighbors == None):
                Removelist.append(T)
                for NeihgborID in T.Neighbors:
                    if NeihgborID != None:
                        for m in range(len(LeftTriangulation[NeihgborID].ListOfVertex)):
                            if not LeftTriangulation[NeihgborID].ListOfVertex[m] in T.ListOfVertex:
                                LeftTriangulation[NeihgborID].Neighbors[m] = None
                        TriangleList = np.concatenate((TriangleList, np.array([LeftTriangulation[NeihgborID]])))
        LeftTriangulation = LeftTriangulation - Removelist
        ###############################################################################################################################################



        #### Order the Neighbors of the trinagle that near the Constraint ####################################################################
        for T in RightTriangulation.Triangles:
            if self.Vertex[constraint[1], :] in RightTriangulation.Vertex[T.ListOfVertex, :] and self.Vertex[constraint[0],:] in RightTriangulation.Vertex[T.ListOfVertex,:]:
                for i in range(len(RightTriangulation.Vertex[T.ListOfVertex, :])):
                    if any(RightTriangulation.Vertex[T.ListOfVertex[i], :] != self.Vertex[constraint[1], :]) and any(RightTriangulation.Vertex[T.ListOfVertex[i], :] != self.Vertex[constraint[0], :]):
                        RigthInx = i
                        RightT = T.ID
                        break

        for T in LeftTriangulation.Triangles:
            if self.Vertex[constraint[1], :] in LeftTriangulation.Vertex[T.ListOfVertex, :] and self.Vertex[constraint[0],:] in LeftTriangulation.Vertex[T.ListOfVertex,:]:
                for i in range(len(LeftTriangulation.Vertex[T.ListOfVertex, :])):
                    if any(LeftTriangulation.Vertex[T.ListOfVertex[i], :] != self.Vertex[constraint[1], :]) and any(LeftTriangulation.Vertex[T.ListOfVertex[i], :] != self.Vertex[constraint[0], :]):
                        LeftInx = i
                        LeftT = T.ID
                        break

        LeftTriangulation[LeftT].Neighbors[LeftInx] = RightT
        RightTriangulation[RightT].Neighbors[RigthInx] = LeftT
        ###############################################################################################################################################

        RightTriangulation.Triangles += LeftTriangulation.Triangles
        RightTriangulation.Vertex[Left,:] = LeftTriangulation.Vertex[Left,:]

        ##### order the Neighbors of the Convex Hull #################################################################################################
        for t in range(len(RightTriangulation.Triangles)):
            if any(RightTriangulation.Triangles[t].Neighbors == None):
                for q in range(len(RightTriangulation.Triangles[t].Neighbors)):
                    if RightTriangulation.Triangles[t].Neighbors[q] == None:
                        myPoints = [RightTriangulation.Triangles[t].ListOfVertex[q-1], RightTriangulation.Triangles[t].ListOfVertex[q-2]]
                        for oldT in Triangle2Remove:
                            if myPoints[0] in self[oldT].ListOfVertex and myPoints[1] in self[oldT].ListOfVertex:
                                for w in range(len(self[oldT].ListOfVertex)):
                                    if self[oldT].ListOfVertex[w] != myPoints[0] and self[oldT].ListOfVertex[w] != myPoints[1] and self[oldT].Neighbors[w] != None:
                                        RightTriangulation.Triangles[t].Neighbors[q] = self[oldT].Neighbors[w]
                                        for u in range(len(self[self[oldT].Neighbors[w]].ListOfVertex)):
                                            if self[self[oldT].Neighbors[w]].ListOfVertex[u] != myPoints[0] and self[self[oldT].Neighbors[w]].ListOfVertex[u] != myPoints[1]:
                                                self[self[oldT].Neighbors[w]].Neighbors[u] = RightTriangulation.Triangles[t].ID
        ###################################################################################################################################################

        ### Conecct the old and the new triangolation + delet the unneccery trialgle ###############################################################################

        self = self - Triangle2Remove

        self.Triangles += RightTriangulation.Triangles

        return self



    def plot(self):
        tri = self.ConvertToarray()
        pnt = self.Vertex
        # Initialize VTK points object
        VtkPnt = vtk.vtkPoints()
        # Initialize color scalars
        pnt_rgb = vtk.vtkUnsignedCharArray()
        # R, G, B
        pnt_rgb.SetNumberOfComponents(3)
        # Colors??
        pnt_rgb.SetName("Colors")

        # Initialize VTK PolyData object for vertices
        vtkVertex = vtk.vtkPolyData()
        # Initialize VTK PolyData object for triangulation
        vtkTri = vtk.vtkPolyData()
        # Initialize VTK vertices object for points
        vtkVertex_ind = vtk.vtkCellArray()
        # Initialize VTK vertices object for triangles
        vtkTri_ind = vtk.vtkCellArray()

        for i in range(len(pnt)):
            # Inserting the i-th point to the vtkPoints object
            id = VtkPnt.InsertNextPoint(pnt[i, 0], pnt[i, 1], pnt[i, 2])
            # Adding the index of i-th point to vertex vtk
            # index array
            vtkVertex_ind.InsertNextCell(1)
            vtkVertex_ind.InsertCellPoint(id)

        # Set vtk point in triangle polydata object
        vtkTri.SetPoints(VtkPnt)
        # Add color to the vtkTri object
        vtkTri.GetPointData().SetScalars(pnt_rgb)
        # Set vtk point in vertexes polydata object
        vtkVertex.SetPoints(VtkPnt)
        vtkVertex.SetVerts(vtkVertex_ind)
        # Add color to the vtkVertex object
        vtkVertex.GetPointData().SetScalars(pnt_rgb)

        for i in range(len(tri)):
            # Set triangles 3 vertices by ID
            ith_tri = vtk.vtkTriangle()
            ith_tri.GetPointIds().SetId(0, int(tri[i, 0]))
            ith_tri.GetPointIds().SetId(1, int(tri[i, 1]))
            ith_tri.GetPointIds().SetId(2, int(tri[i, 2]))
            # Insert the i-th triangle data index
            vtkTri_ind.InsertNextCell(ith_tri)

        # Initialize a VTK mapper
        vtkMapper = vtk.vtkPolyDataMapper()
        vtkMapper.SetInputData(vtkVertex)
        vtkTri.SetPolys(vtkTri_ind)
        vtkMapper.SetInputData(vtkTri)
        # Initialize a VTK actor
        vtkActor = vtk.vtkActor()
        vtkActor.SetMapper(vtkMapper)
        # Initialize a VTK render window
        vtkRenderWindow = vtk.vtkRenderWindow()

        # Initialize a VTK renderer.
        # Contains the actors to render
        vtkRenderer = vtk.vtkRenderer()
        # Add the VTK renderer to the VTK render window
        vtkRenderWindow.AddRenderer(vtkRenderer)
        # define the renderer.
        vtkRenderer.AddActor(vtkActor)

        vtkActor.GetProperty().SetRepresentationToWireframe()

        # Set camera and background data
        vtkRenderer.ResetCamera()
        vtkRenderWindow.Render()
        # Enable user interface interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(vtkRenderWindow)
        vtkRenderWindow.Render()
        interactor.Start()





