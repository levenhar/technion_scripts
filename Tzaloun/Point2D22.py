'''
This class gets values and make them to a point from Points2D type.
'''
import math
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


class Point2D:
    def __init__(self,X,Y,ID,H=None,Sxx=0,Syy=0,Sxy=0,Sh=0):
        self._x=X
        self._y=Y
        self._H=H
        self._id=ID
        self.Sxx=Sxx
        self.Syy = Syy
        self.Sxy = Sxy
        self.Sh=Sh
        self.Smax=0
        self.Smin=0
        self.phi=0

    @property
    def my_h(self):
        return self._H

    @my_h.setter
    def my_h(self, value):
        if type(value) is float:
            self._H = value
        else:
            print("error")


    '''making sure that X and Y will be a float type:'''
    @property
    def my_x(self):
        return self._x
    @my_x.setter
    def my_x(self,value):
        if type(value) is float:
            self._x= value
        else:
            print("error")

    @property
    def my_y(self):
        return self._y
    @my_y.setter
    def my_y(self, value):
        if type(value) is float:
            self._y = value
        else:
            print("error")

    '''making sure that ID will be a int type:'''
    @property
    def my_id(self):
        return self._id

    @my_id.setter
    def my_id(self, value):
        if type(value) is int:
            self._id = value
        else:
            print("error")

    '''The way we want our print will be:'''
    def __repr__(self):
        return "id="+str(self._id)+ " (x=" +str(round(self._x,3)) + ", y=" +str(round(self._y,3)) + ")"

    #check if two points are equal or if the string is the name of the point
    def __eq__(self, other):
        if type(other) == Point2D:
            if round(self.my_x,5)==round(other.my_x,5) and round(self.my_y,5)==round(other.my_y,5):
                return True
            else:
                return False
        elif type(other) == type(""):
            if self.my_id == other:
                return True
            else:
                return False

    #plot a point
    def plot(self,ax,type="TS"):
        if type=="TS":
            ax.scatter([self.my_y],[self.my_x],color="black",marker="^")
            ax.annotate(f"{self.my_id}",(self.my_y+5, self.my_x+5))
            if self.Smax and self.Smin:
                ellipse = Ellipse(xy=(self.my_y,self.my_x), width=self.Smin, height=self.Smax,edgecolor='r',angle=self.phi, fc='None', lw=2)
                ax.add_patch(ellipse)
        elif type=="Prat":
            ax.scatter([self.my_y], [self.my_x], [2], color="black")
            #ax.annotate(f"{self.my_id}", (self.my_y + 2, self.my_x + 2))


    def azimuth(self,p2):
        dx=p2.my_x-self.my_x
        dy=p2.my_y-self.my_y
        if dx == 0:
            if dy>=0:
                return 0
            else:
                return 180
        elif dy==0:
            if dx>0:
                return 90
            else:
                return 270
        else:
            a=(180/math.pi)*math.atan(abs(dy/dx))
            if dy>0 and dx>0:
                return a
            elif dy>0 and dx<0:
                return 180-a
            elif dy<0 and dx>0:
                return 360-a
            else:
                return 180+a

    def ErrorElipse(self):
        t=math.sqrt((self.Sxx-self.Syy)**2+4*self.Sxy**2)
        self.Smax=math.sqrt(0.5*(self.Sxx+self.Syy+t))
        self.Smin = math.sqrt(0.5 * (self.Sxx + self.Syy - t))
        self.phi=0.5*math.atan2((2*self.Sxy),(self.Sxx-self.Syy))


