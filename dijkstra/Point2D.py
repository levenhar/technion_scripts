class Point2D:
    def __init__(self,x,y,id):
        '''
        :param x: x value
        :type x: float
        :param y: y value
        :type y: float
        :param id: point ID
        :type id: string
        '''
        self.X=x
        self.Y=y
        self.ID=id

    @property
    def X(self):
        return self._x
    @X.setter
    def X(self,X):
        if isinstance(X,float) or isinstance(X,int):
            self._x=X
        else:
            raise Exception("x and y value need to be a float value")

    @property
    def Y(self):
        return self._y
    @Y.setter
    def Y(self,Y):
        if isinstance(Y, float) or isinstance(Y, int):
            self._y=Y
        else:
            raise Exception("x and y value need to be a float value")

    @property
    def ID(self):
        return self._id
    @ID.setter
    def ID(self,ID):
        self._id = str(ID)

    #print function
    def __repr__(self):
        return f"id={self._id},(X={self._x},Y={self._y})"

    def distance_to_point(self,point):
        '''
        calculate the distance between self to other point.
        :param point:the other point
        :type point:Point2D
        :return: the distance
        :rtype:float
        '''
        return round((((self._x-point._x)**2+(self._y-point._y)**2)**0.5),3)

    def point_translate(self,dx,dy):
        '''
        moving the point
        :param dx: delta x value
        :type dx: float
        :param dy: delta y value
        :type dy: float
        :return: none
        :rtype:none
        '''
        self._x += dx
        self._y +=dy

    def distance_statics(self, Pnt1, Pnt2):
        '''
        calculate distance between two points
        :param Pnt1: first point
        :type Pnt1: Point2D
        :param Pnt2: second point
        :type Pnt2: Point2D
        :return: the distance
        :rtype: float
        '''
        return round((((Pnt1._x - Pnt2._x) ** 2 + (Pnt1._y - Pnt2._y) ** 2) ** 0.5), 3)
