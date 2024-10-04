from tkinter import *
from tkinter import messagebox
import time
import pandas as pd
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from tkinter import simpledialog
import numpy as np
from Kd_tree import kd_tree
from EqualCells import EqualCells
from PointsCloud import PointsCloud
from more_itertools import partition



class UI:
    def __init__(self,root):
        self.top = root  #main window of GUI
        self.win_DBType= None  #secondry window for choching DB type
        self.DataBase = "" #pd.dataframe that contain the data
        self.Cloud = None #Points Cloud that contain the data
        self.is_DB = False #mark if the database had loaded succeessfully
        self.DataBaseType = None # mark the DB type that chosen

        self.n0 = 0 #the averge point in cell for equal area DB

        self.nMAX = 0 #the maximum point in one cell for Quadtree
        self.tree = None #the Quadtree database

        self.fig = Figure(figsize=(6, 6), dpi=100, )  # Creating a figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)  # A tk.DrawingArea.
        self.toolbar_frame = Frame(self.top)  # Creating a toolbar
        self.ax = self.fig.add_subplot(111, projection="3d")  # Create the 3D axis
        self.show_toolbar = True #for creating the toolbar only in the first time.
        self.is_plot = False # mark if there is a plot all ready
        self.is_selection = False #mark if the user already make a selection

        self.ground_list = PointsCloud([]) #list of all the points on the ground
        self.objects_list = PointsCloud([])

    def ui(self):
        # Place the Buttons on the screen
        Button(text="Import CSV file", command=self.loading_CSV,master=self.top).grid(row=0, column=0)
        Button(text="filter Points cloud", command=self.filtering,master=self.top).grid(row=0, column=3)
        Button(text="Clear filtered points", command=self.clear_selection, master=self.top).grid(row=0, column=4)
        Button(text="Clear DB", command=self.clear_DB, master=self.top).grid(row=0, column=5)
        Button(text="Quit", command=self.Quit,master=self.top).grid(row=0, column=6)


    def loading_CSV(self):
        '''
        Function that load CSV file as DataFrame, and select the min and max coordinate in the whole database.
        At least this function call to chosing_DBType function.
        '''
        if not self.is_plot:
            self.top.filename = filedialog.askopenfilename(initialdir="/", title="select vertices csvfile",
                                                           filetypes=(("xyzfiles", " *.xyz"),))
            self.DataBase = pd.read_csv(self.top.filename, sep='\s+', header=None, names=['x', 'y', 'z'])

            self.DataBase=self.DataBase.to_numpy().astype(np.float)

            self.is_DB = True
            self.Cloud = PointsCloud(self.DataBase)

            self.chosing_DBType()




        else:
            messagebox.showinfo('Error', 'Please clear the plot before loading new database')


    def chosing_DBType(self):
        '''
        Function that create new window and place in it 3 buttons for choosing database type.
        '''
        self.win_DBType = Tk()
        self.win_DBType.title('choosing database Type')
        self.win_DBType.geometry('300x50+0+0')
        Button(text="Basic DB", command=self.basic, master=self.win_DBType).grid(row=0, column=0)
        Button(text="Equal Area DB", command=self.byArea,master=self.win_DBType).grid(row=0, column=1)
        Button(text="k-d tree DB", command=self.KDtree, master=self.win_DBType).grid(row=0, column=2)
        self.win_DBType.mainloop()


    def basic(self):
        '''
        Mark the database type as basic, close the small window and call to ploting function
        '''
        self.DataBaseType = "Basic"

        self.win_DBType.quit()
        self.win_DBType.destroy()
        self.ploting()


    def byArea(self):
        '''
        Mark the database type as byArea.
        The function creat the database matrix and calculate the value that define it.
        At least it close the small window and call to ploting function
        '''
        self.DataBaseType="byArea"
        self.n0 = simpledialog.askinteger("average point in a cell", "please input the average points in one cell",parent=self.top)

        self.cells=EqualCells(self.Cloud,self.n0)

        self.cells.DBMatrix=np.array(self.cells.DBMatrix,dtype=object)

        self.win_DBType.quit()
        self.win_DBType.destroy()
        self.ploting()


    def KDtree(self):
        '''
        Mark the database type as kd-tree and create the kd-tree (by calling to the class) according to the maximum point in cell that it get from the user.
        ALso it close the small window and call to ploting function
        '''
        self.DataBaseType="k-d tree"
        self.nMAX = 10

        self.tree = kd_tree(self.nMAX,self.Cloud,0)

        self.win_DBType.quit()
        self.win_DBType.destroy()
        self.ploting()

    def ploting(self):
        '''
        Functin that printing the points, and print the database form according to the type that selected by the user.
        :return:
        :rtype:
        '''
        if self.DataBaseType != None:
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=10)
            self.ax.scatter(self.Cloud["x"], self.Cloud["y"], self.Cloud["z"], color='k', s=0.1)# plot all the points
            self.canvas.draw()
            #plt.show()

            self.is_plot=True
            if self.show_toolbar:  # Display the toolbar
                self.toolbar_frame.grid(row=3, column=0, columnspan=9)
                self.navigation = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
                self.navigation.update()
                self.show_toolbar = False
            self.canvas.draw()
        else:
            messagebox.showinfo('Error', 'Please chose DB type')



    def Search(self,p0,R):
        '''
        search the points in the searching area, in 3 different ways, according to the DB type
        '''
        selection_list = PointsCloud([])
        R2=R**2
        if self.is_DB:
            #p0 = self.Cloud[inx]
            if self.DataBaseType =='Basic':
                selection_list = selection_list + list(filter(lambda P: P.Square_distance_to_point(p0) <= R2, self.Cloud.pointsC))
            elif self.DataBaseType == "byArea":
                Ymin1 = p0.Y - R
                Ymax1 = p0.Y + R
                Xmin1 = p0.X - R
                Xmax1 = p0.X + R
                Zmin1 = p0.Z - R
                Zmax1 = p0.Z + R
                n_up_rightY, n_up_rightX, n_up_rightZ = self.cells.XYZ2IJK(Ymax1,Xmax1,Zmax1)
                n_down_leftY, n_down_leftX, n_down_leftZ = self.cells.XYZ2IJK(Ymin1,Xmin1, Zmin1)
                for i in range(n_down_leftY-1, n_up_rightY+1): #loop that check all of the points in the overlap cells with the searching area
                    for j in range(n_down_leftX-1, n_up_rightX+1):
                        for k in range(n_down_leftZ-1, n_up_rightZ+1):
                            distanceList=list(map(lambda P:self.Cloud[P].Square_distance_to_point(p0),self.cells.DBMatrix[i,j,k]))
                            for m in range(len(distanceList)):
                                if distanceList[m] <= R2:
                                    selection_list = selection_list + (self.Cloud[self.cells.DBMatrix[i,j,k][m]])
            else:  #for k-d tree
                bound = [p0.Y - R, p0.Y + R, p0.X - R, p0.X + R, p0.Z - R, p0.Z + R]
                selection_list=self.tree.searchInTree(bound,p0,R,selection_list)
            return selection_list
        else:
            messagebox.showinfo('Error', 'Please load DB file')


    def grounddetector(self,P0):
        selction=self.Search(P0,self.R)
        if min(selction["z"]) == P0["z"]:
            return 1
        else:
            #P0=self.Cloud[inxx]
            slopeList=list(map(lambda P:abs(P0["z"]-P["z"])/P0.Square_distance_to_point(P)**0.5 if P["z"] < P0["z"] else 0, selction))
            if any(slopeList>self.Slope_t):
                return 0
            else:
                return 1


    def filtering(self):
        self.R = simpledialog.askfloat("radius", "please input the radius",parent=self.top)
        self.Slope = simpledialog.askfloat("slope", "please input the maximum slope",parent=self.top)
        self.Slope_t = np.arctan(self.Slope*np.pi/180)
        start_time = time.time()
        O, G = partition(lambda P: self.grounddetector(P), self.Cloud)
        self.ground_list = self.ground_list + list(G)
        self.objects_list = self.objects_list + list(O)
        end_time = time.time()

        self.FilteringTime = round(end_time - start_time, 3)
        messagebox.showinfo('Filtering Time', f'Filtering Time: {self.FilteringTime}')

        self.plot_Selection_Points()


    def clear_DB(self):
        '''
        function that delet all of the DB setting and data.
        '''
        if self.is_DB:
            self.ax.clear()
            self.canvas.draw()
            self.DataBaseType = None
            self.is_plot = False
            self.is_DB = False
            self.ground_list = PointsCloud([])
            self.objects_list = PointsCloud([])
            self.is_selection = False
            try:
                self.Ground_win.destroy()
            except:
                pass
            try:
                self.object_win.destroy()
            except:
                pass
            return


        else:
            messagebox.showinfo('Error', 'Please load DB file')


    def clear_selection(self):
        '''
        function that delete the plot and draw only the point and the cells.
        Also it deleting the selection information.
        '''
        if self.is_DB:
            self.ax.clear()
            self.canvas.draw()
            self.ploting()
            self.ground_list = PointsCloud([])
            self.objects_list = PointsCloud([])
            self.is_selection = False
            try:
                self.Ground_win.destroy()
            except:
                pass
            try:
                self.object_win.destroy()
            except:
                pass

            return
        else:
            messagebox.showinfo('Error', 'Please load DB file')



    def plot_Selection_Points(self):
        '''
        Plot in a new color, the selected points
        '''
        self.ax.clear()
        self.canvas.draw()
        XX = list(map(lambda P: P.X, self.ground_list))
        YY = list(map(lambda P: P.Y, self.ground_list))
        ZZ = list(map(lambda P: P.Z, self.ground_list))
        self.ax.scatter(XX, YY, ZZ, color='g', s=2)
        self.canvas.draw()

        self.Ground_win = Tk()
        self.Ground_win.title('Ground points display')
        fig1 = Figure(figsize=(6, 6), dpi=100, )  # Creating a figure
        canvas1 = FigureCanvasTkAgg(fig1, master=self.Ground_win)  # A tk.DrawingArea.
        ax1 = fig1.add_subplot(111, projection="3d")
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=2, column=0, columnspan=10)
        ax1.scatter(self.ground_list["x"], self.ground_list["y"], self.ground_list["z"], color='g', s=0.1)  # plot all the points
        canvas1.draw()

        self.object_win = Tk()
        self.object_win.title('Object points display')
        fig2 = Figure(figsize=(6, 6), dpi=100, )  # Creating a figure
        canvas2 = FigureCanvasTkAgg(fig2, master=self.object_win)  # A tk.DrawingArea.
        ax2 = fig2.add_subplot(111, projection="3d")
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=2, column=0, columnspan=10)
        XX = list(map(lambda P: P.X , self.objects_list))
        YY = list(map(lambda P: P.Y , self.objects_list))
        ZZ = list(map(lambda P: P.Z , self.objects_list))

        ax2.scatter(XX, YY, ZZ, color='r',s=0.1)  # plot all the points
        canvas2.draw()

        self.ax.scatter(XX, YY, ZZ, color='r',s=0.1)
        self.canvas.draw()


        my_string=f"The scanning area box -\n" \
                  f"x:({min(self.Cloud['x'])},{max(self.Cloud['x'])}) \n" \
                  f"y:({min(self.Cloud['y'])},{max(self.Cloud['y'])}) \n" \
                  f"z:({min(self.Cloud['z'])},{max(self.Cloud['z'])}) \n" \
                  f"\n \n \n" \
                  f"Filter Parameters - Search Radius: {self.R}, Maximum Slope: {self.Slope}°" \
                  f"\nNumber of points (total) = {len(self.Cloud)}" \
                  f"\nNumber of ground points = {len(self.ground_list)}" \
                  f"\nFiltered points (percent) = {round((len(self.Cloud)-len(self.ground_list))/len(self.ground_list)*100,3)}%" \
                  f"\n \n" \
                  f"Scanning Time = {self.FilteringTime} [sec]" \
                  f"\n" \
                  f"Data Base type - {self.DataBaseType}"
        if self.n0 != 0:
            my_string += f"\n \nAverage number of point in cell = {self.n0}"

        with open(f"{self.top.filename[-9:-4]}_type-{self.DataBaseType}_R-{self.R}[m]_Slope-{self.Slope}°.txt", "w") as file:
            file.write(my_string)

    def Quit(self):  # Close the program
        self.top.quit()
        self.top.destroy()

