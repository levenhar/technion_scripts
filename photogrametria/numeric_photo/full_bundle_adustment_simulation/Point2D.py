from math import sqrt
import numpy as np

class ImagePoints:
    def __init__(self,ID, x, y, type, Control=None):  # Define the class fields with a default value for id
        """x- the x value of the point
        y- the y value of the point"""
        self.x = x
        self.y = y
        self.Images = []
        self.type = type
        self.groundPoint = Control
        self.id=ID


    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        self.__y = value

    def distance_to_point(self, point):  # Calculate the distance between the points
        return sqrt((self.__x - point.x) ** 2 + (self.__y - point.y) ** 2)

    @staticmethod
    def distance(point1, point2):  # Calculate the distance between two points, point1 and point2
        return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


    def __eq__(self, other):
        if isinstance(other,ImagePoints):
            if self.id==other.id:
                return True
            else:
                return False
        elif isinstance(other, str):
            if self.id==other:
                return True
            else:
                return False


    def convert2list(self):
        return [self.id,str(self.x), str(self.y)]


    def AddNoise(self, r):
        noise = r * np.random.random(2)
        self.x += noise[0]
        self.y += noise[1]

    @staticmethod
    def listofPoints2Array(Pointslist):
        PointsArr=np.array(list(map(lambda P: P.convert2list() if P != None else np.array([None,None, None]), Pointslist)))
        return PointsArr