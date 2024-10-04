import pandas as pd
from Point2D import Point2D


class quadtree:
    def __init__(self,n,pointsList,RightUpE,RightUpN,LeftDownE,LeftDownN,ax,canvas):
        self.ax=ax #save the axis
        self.canvas=canvas #save the canvas
        self.nMax=n #the maximum point in cell
        self.RightUpE = RightUpE
        self.LeftDownE = LeftDownE
        self.RightUpN = RightUpN
        self.LeftDownN = LeftDownN
        self.Sons = [] #List of all the sons
        self.points = pointsList #DataFrame of all the points
        '''
        the recursion part, if there is more than n point in a cell, it split him to 4 new son
        '''
        if len(self.points[0])>=self.nMax:
            list1 = pd.DataFrame([],columns=[0,1])
            list2 = pd.DataFrame([],columns=[0,1])
            list3 = pd.DataFrame([],columns=[0,1])
            list4 = pd.DataFrame([],columns=[0,1])
            for i in range(len(self.points[0])): #Loop that conect between the points to the relevant son.
                if ((self.RightUpE+self.LeftDownE)/2) <= self.points[0][i] <= self.RightUpE and ((self.RightUpN+self.LeftDownN)/2) <= self.points[1][i] <= self.RightUpN:
                    list1.loc[len(list1)] = [self.points[0][i],self.points[1][i]]
                elif self.LeftDownE <= self.points[0][i] <= ((self.RightUpE+self.LeftDownE)/2) and ((self.RightUpN+self.LeftDownN)/2) <= self.points[1][i] <=self.RightUpN:
                    list2.loc[len(list2)] = [self.points[0][i],self.points[1][i]]
                elif self.LeftDownE <= self.points[0][i] <= ((self.RightUpE+self.LeftDownE)/2) and self.LeftDownN <= self.points[1][i] <=((self.RightUpN+self.LeftDownN)/2):
                    list3.loc[len(list3)] = [self.points[0][i],self.points[1][i]]
                else:
                    list4.loc[len(list4)] = [self.points[0][i],self.points[1][i]]

            #Creating the sons
            son1 = quadtree(self.nMax,list1,self.RightUpE,self.RightUpN,(self.RightUpE+self.LeftDownE)/2,(self.RightUpN+self.LeftDownN)/2,self.ax,self.canvas)
            son2 = quadtree(self.nMax,list2,(self.RightUpE+self.LeftDownE)/2,self.RightUpN,self.LeftDownE,(self.RightUpN+self.LeftDownN)/2,self.ax,self.canvas)
            son3 = quadtree(self.nMax,list3,(self.RightUpE+self.LeftDownE)/2,(self.RightUpN+self.LeftDownN)/2,self.LeftDownE,self.LeftDownN,self.ax,self.canvas)
            son4 = quadtree(self.nMax,list4,self.RightUpE,(self.RightUpN+self.LeftDownN)/2,(self.RightUpE+self.LeftDownE)/2,self.LeftDownN,self.ax,self.canvas)

            #Creating the sons list
            self.Sons.append(son1)
            self.Sons.append(son2)
            self.Sons.append(son3)
            self.Sons.append(son4)
            self.points = [] #Deleting the points list from the parent.


    def overlap(self,E_UR_2,N_UR_2,E_DL_2,N_DL_2):
        '''
        :param E_UR_2:  E of the up right corner of the second rectangle
        :param N_UR_2:  N of the up right corner of the second rectangle
        :param E_DL_2:  E of the down left corner of the second rectangle
        :param N_DL_2:  N of the down left corner of the second rectangle

        Function that check if a rectangle overlap with the cell himself
        '''
        maxN = max(self.LeftDownN,N_DL_2)
        minN = min(self.RightUpN,N_UR_2)
        maxE = max(self.LeftDownE,E_DL_2)
        minE = min(self.RightUpE,E_UR_2)
        if minN >= maxN and minE >= maxE:
            return True
        return False

    def PlotBorder(self):
        #ploting the border of the son celles
        if self.Sons:
            self.ax.plot([self.LeftDownE, self.RightUpE],[(self.RightUpN+self.LeftDownN)/2, (self.RightUpN+self.LeftDownN)/2], color='red', linewidth='0.5')
            self.ax.plot([(self.RightUpE+self.LeftDownE)/2, (self.RightUpE+self.LeftDownE)/2], [self.LeftDownN, self.RightUpN], color='red', linewidth='0.5')
            self.canvas.draw()
            self.Sons[0].PlotBorder()
            self.Sons[1].PlotBorder()
            self.Sons[2].PlotBorder()
            self.Sons[3].PlotBorder()

    def searchInTree(self,Nmin1,Nmax1,Emin1,Emax1,p0,R,selectlist):
        '''
        :param p0: center of the blocking rectangle
        :param R:radius of searching
        :param selectlist: list of the selected points

        Function that search in the quadtree, by recursion.
        if overlap and there is not sons, the function check the if the point in the point list placed inside the searching area.

        '''
        if self.overlap(Emax1,Nmax1,Emin1,Nmin1):
            if not self.Sons:
                for i in range(len(self.points)):
                    point1=Point2D(self.points[0][i],self.points[1][i])
                    if point1.Square_distance_to_point(p0) <= R ** 2:
                        selectlist.append((float(self.points[0][i]), float(self.points[1][i])))
            else:
                for S in self.Sons:
                    S.searchInTree(Nmin1,Nmax1,Emin1,Emax1,p0,R,selectlist)
        return selectlist



