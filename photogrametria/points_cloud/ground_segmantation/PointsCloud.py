import numpy as np
from Point3D import Point3D

class PointsCloud:
    def __init__(self,Points):

        self.pointsC=[]
        if Points != []:
            self.pointsC=list(map(lambda x,y,z:Point3D(float(x), float(y),float(z)),Points[:,0],Points[:,1],Points[:,2]))

    def __add__(self, other):
        if isinstance(other, Point3D):
            self.pointsC.append(other)
            return self
        elif isinstance(other, list):
            self.pointsC+=other
            return self
        elif isinstance(other, np.ndarray):
            self.pointsC.append(Point3D(float(other[0]), float(other[1]),float(other[2])))
            return self
        else:
            raise Exception("can add only Point3D, list of Points or array")

    def __getitem__(self, item):
        if isinstance(item,int):
            return self.pointsC[item]
        else:
            ItemList=list(map(lambda P: P[item],self.pointsC))
            return ItemList

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self.pointsC):
            result = self.pointsC[self.current]
            self.current += 1
            return result
        else:
            raise StopIteration


    def __len__(self):
        return len(self.pointsC)


    def __sub__(self, other):
        if isinstance(other,Point3D):
            listt=list(filter(lambda P:P != other, self))
            return listt
        elif isinstance(other, PointsCloud):
            listt=list(filter(lambda P:P if P not in other else 0, self))
            return listt
        else:
            raise Exception("this operate work on Point3D or Pointscloud only")
