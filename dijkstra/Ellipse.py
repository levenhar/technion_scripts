import Point2D
import numpy as np

class Ellipse():
    ''' create an ellipse'''
    def __init__(self,center=Point2D.Point2D(0, 0, "Unit Ellipse"), a=1.0, b=1.0, name="Unit Ellipse",x=None,y=None):
        '''
        Ellipse creator.
        for use Center as X and Y value, you need to write before every argument which variable it belongs
        for example: Ellipse(x=1,y=1, a=4,b=8,name="E1").

        :param center: the center of the ellipse
        :type center: Point2D
        :param a: axis x radios
        :type a: float
        :param b: axis Y radios
        :type b: float
        :param name: an ID
        :type name: string
        '''
        if x==None:
            self.Center = center
        else:
            self.Center = Point2D.Point2D(x,y,f"CenterOf{name}")
        self._name = str(name)
        self.A = a
        self.B = b

    @property
    def Center(self):
        return self._Center
    @Center.setter
    def Center(self,point):
        self._Center = point

    @property
    def A(self):
        return self._a
    @A.setter
    def A(self, a):
        if a != 0:
            try:
                self._a = abs(float(a))
            except ValueError:
                print("a value is illegal")
                raise Exception("pleas enter a float number")
        else:
            raise Exception("the radios can not be equal to Zero")

    @property
    def B(self):
        return self._b
    @B.setter
    def B(self, b):
        if  b != 0:
            try:
                self._b = abs(float(b))
            except ValueError:
                print(" b value is illegal")
                raise Exception("pleas enter a float number")
        else:
            raise Exception("the radios can not be equal to Zero")

    @property
    def area(self):
        area = round(np.pi*self._a*self._b, 3)
        return area

    #print function
    def __repr__(self):
        return f"{self._name}: X={self._Center._x}, Y={self._Center._y}, a={self._a}, b={self._b}, Area={self.area}  "

    def transelate_center(self,dx,dy):
        '''
        function that move the center of an ellipse.
        :param dx: delta x value
        :type dx: float
        :param dy: delta y value
        :type dy: float
        '''
        self._Center.point_translate(dx,dy)

