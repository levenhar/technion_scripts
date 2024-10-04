

class Point3D:
    def __init__(self, x, y, z):
        '''
        :param x: x value
        :type x: float
        :param y: y value
        :type y: float
        :param z: z value
        :type z: float

        '''
        self._x = x
        self._y = y
        self._z = z

    @property
    def Axis(self):
        return self._axis
    @Axis.setter
    def X(self, ax):
        if ax=="x" or ax=="y" or ax=="z":
            self._axis = ax
        else:
            raise Exception("axis value can be \"x\", \"y\" or \"z\"")

    @property
    def X(self):
        return self._x
    @X.setter
    def X(self,X):
        if isinstance(X,float) or isinstance(X,int):
            self._x=X
        else:
            raise Exception("x, y and z value need to be a float value")

    @property
    def Y(self):
        return self._y
    @Y.setter
    def Y(self,Y):
        if isinstance(Y, float) or isinstance(Y, int):
            self._y=Y
        else:
            raise Exception("x, y and z value need to be a float value")

    @property
    def Z(self):
        return self._z

    @Z.setter
    def Z(self, Z):
        if isinstance(Z, float) or isinstance(Z, int):
            self._z = Z
        else:
            raise Exception("x, y and z value need to be a float value")


    #print function
    def __repr__(self):
        return f"(X={self._x},Y={self._y}, Z={self._z})"

    def distance_to_point(self,point):
        '''
        calculate the distance between self to other point.
        :param point:the other point
        :type point:Point3D
        :return: the distance
        :rtype:float
        '''
        return round((((self._x-point.X)**2+(self._y-point.Y)**2+(self._z-point.Z)**2)**0.5),3)

    def Square_distance_to_point(self,point):
        return (self._x-point.X)**2+(self._y-point.Y)**2+(self._z-point.Z)**2


    def __getitem__(self, item):
        if item=="x" or item=="X":
            return self.X
        elif item=="y" or item=="Y":
            return self.Y
        elif item=="z" or item=="Z":
            return self.Z
        else:
            raise Exception("The value can be \"x\", \"y\" or \"z\"")

    def convert2list(self):
        return [round(self._x,5), round(self._y,5), round(self._z,5)]

    def __eq__(self, other):
        if self["x"]==other["x"] and self["y"]==other["y"] and self["z"]==other["z"]:
            return True
        return False
