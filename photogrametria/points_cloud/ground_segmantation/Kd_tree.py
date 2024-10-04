
import numpy as np
from PointsCloud import PointsCloud


class kd_tree:
    def __init__(self,n,pointsList,level):
        self.nMax=n #the maximum point in cell
        self.level=level
        self.Xmin = min(pointsList["x"])
        self.Ymin = min(pointsList["y"])
        self.Zmin = min(pointsList["z"])
        self.Xmax = max(pointsList["x"])
        self.Ymax = max(pointsList["y"])
        self.Zmax = max(pointsList["z"])
        self.mid = 0
        self.Sons = [] #List of all the sons
        self.points = pointsList #Points Cloud of all the points

        '''
        the recursion part, if there is more than n point in a cell, it split him to 2 new son
        '''
        if len(self.points)>=self.nMax:
            list1 = PointsCloud([])
            list2 = PointsCloud([])
            # Loop that connect between the points to the relevant son.
            #each time split by diffrent axis
            if self.level % 3 == 0:
                self.mid=np.median(pointsList["x"])
                for i in range(len(self.points)):
                    if self.points[i].X >= self.mid:
                        list1+(self.points[i])
                    else:
                        list2+(self.points[i])
            elif self.level % 3 == 1:
                self.mid=np.median(pointsList["y"])
                for i in range(len(self.points)):
                    if self.points[i].Y >= self.mid:
                        list1+(self.points[i])
                    else:
                        list2+(self.points[i])
            elif self.level % 3 == 2:
                self.mid=np.median(pointsList["z"])
                for i in range(len(self.points)):
                    if self.points[i].Z >= self.mid:
                        list1+(self.points[i])
                    else:
                        list2+(self.points[i])

            #Creating the sons
            son1 = kd_tree(self.nMax,list1,self.level+1)
            son2 = kd_tree(self.nMax,list2,self.level+1)


            #Creating the sons list
            self.Sons.append(son1)
            self.Sons.append(son2)

            self.points = [] #Deleting the points list from the parent.


    def overlap(self,bound):
        '''

        Function that check if a rectangle overlap with the cell himself
        '''

        maxY = max(self.Ymin, bound[0])
        minY = min(self.Ymax, bound[1])
        maxX = max(self.Xmin, bound[2])
        minX = min(self.Xmax, bound[3])
        maxZ = max(self.Zmin, bound[4])
        minZ = min(self.Zmax, bound[5])
        if minY >= maxY and minX >= maxX and minZ >= maxZ:
            return True
        return False

    def searchInTree(self,bound,p0,R,selectlist):
        '''
        :param p0: center of the blocking rectangle
        :param R:radius of searching
        :param selectlist: list of the selected points

        Function that search in the quadtree, by recursion.
        if overlap and there is not sons, the function check the if the point in the point list placed inside the searching area.

        '''

        R2 = R ** 2
        if self.overlap(bound):
            if not self.Sons:
                selectlist = selectlist + list(filter(lambda P: P.Square_distance_to_point(p0)<=R2, self.points))
            else:
                for S in self.Sons:
                    S.searchInTree(bound, p0, R, selectlist)
        return selectlist



