from tkinter import *
from tkinter import messagebox
import psycopg2 as pg
import time
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from tkinter import simpledialog
import numpy as np
from Point2D import Point2D
import matplotlib.pyplot as plt
from Quadtree import quadtree
import matplotlib.patches as patches


class UI:
    def __init__(self,root):
        self.top = root  #main window of GUI
        self.win_DBType= None  #secondry window for choching DB type
        self.win_PointsTable = None #secondry window for display the selected points
        self.DataBase = "" #pd.dataframe that contain the data
        self.is_DB = False #mark if the database had loaded succeessfully
        self.DataBaseType = None # mark the DB type that chosen
        self.n0 = 0 #the averge point in cell for equal area DB
        self.ratio = 0 #the ratio between the rectangle sides for equal area DB
        self.b = 0 # length of E side in one cell for equal area DB
        self.a = 0 #length of N side in one cell for equal area DB
        self.nMAX = 0 #the maximum point in one cell for Quadtree
        self.tree = None #the Quadtree database
        self.fig = Figure(figsize=(6, 6), dpi=100, )  # Creating a figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)  # A tk.DrawingArea.
        self.toolbar_frame = Frame(self.top)  # Creating a toolbar
        self.Nmin = 0 #The minimun N coordinate of the whole database
        self.Nmax = 0 #The maximum N coordinate of the whole database
        self.Emin = 0 #The minimun E coordinate of the whole database
        self.Emax = 0 #The maximum E coordinate of the whole database
        self.ax = self.fig.add_subplot(111)  # Create the axis
        self.ax.axis('equal')
        self.show_toolbar = True #for creating the toolbar only in the first time.
        self.is_plot = False # mark if there is a plot all ready
        self.is_selection = False #mark if the user already make a selection
        self.SearchE = 0 # the E coordinate of the center of the search area
        self.SearchN = 0 #the N coordinate of the center of the search area
        self.SearchR = 0 #the radius of the search area
        self.selection_list = [] #list of all the selected points
        self.selection_Table = ttk.Treeview() #the object that cuntain the seleced point table
        self.clicks = 0 #mark how much times the user click on the roller

    def ui(self):
        # Place the Buttons on the screen
        Button(text="Import CSV file", command=self.loading_CSV,master=self.top).grid(row=0, column=0)
        Button(text="Import SQL file", command=self.loading_SQL, master=self.top).grid(row=0, column=1)
        Button(text="Add coordinates and radius of searching", command=self.Add_Coor_R, master=self.top).grid(row=0, column=2)
        Button(text="Search", command=self.Search,master=self.top).grid(row=0, column=3)
        Button(text="Clear selection", command=self.clear_selection, master=self.top).grid(row=0, column=4)
        Button(text="Clear DB", command=self.clear_DB, master=self.top).grid(row=0, column=5)
        Button(text="Quit", command=self.Quit,master=self.top).grid(row=0, column=6)


    def loading_CSV(self):
        '''
        Function that load CSV file as DataFrame, and select the min and max coordinate in the whole database.
        At least this function call to chosing_DBType function.
        '''
        if not self.is_plot:
            self.top.filename = filedialog.askopenfilename(initialdir="/", title="select vertices csvfile",
                                                           filetypes=(("csvfiles", " *.csv"),))
            self.DataBase = pd.read_csv(self.top.filename, header=None, usecols=[0,1])
            try:
                float(self.DataBase[0][0])
                float(self.DataBase[1][0])
                self.is_DB = True


                self.Nmin = min(self.DataBase[1])
                self.Emin = min(self.DataBase[0])
                self.Nmax = max(self.DataBase[1])
                self.Emax = max(self.DataBase[0])

                self.chosing_DBType()
            except ValueError:
                messagebox.showinfo('Error', 'Please check that there is not heading in the CSV file')

        else:
            messagebox.showinfo('Error', 'Please clear the plot before loading new database')


    def loading_SQL(self):
        '''
        Function that load SQL file as DataFrame, and select the min and max coordinate in the whole database.
        At least this function call to chosing_DBType function.

        For this home work, the database name is - Big
        and all other parameter is given as default.
        '''
        if not self.is_plot:
            DBname = simpledialog.askstring("SQL file Name", "please enter the SQL file name",parent=self.top)

            try:
                conn = pg.connect(
                    user="postgres",
                    password="mortchelet",
                    host="127.0.0.1",
                    port="5432",
                    database=DBname)

                curs = conn.cursor()
                PointsQuery = 'SELECT \"E\",\"N\" FROM \"Points\"'
                curs.execute(PointsQuery)
                DataBase = curs.fetchall()
                self.DataBase = pd.DataFrame(DataBase,columns=[0,1])
                self.is_DB = True


                self.Nmin = min(self.DataBase[1])
                self.Emin = min(self.DataBase[0])
                self.Nmax = max(self.DataBase[1])
                self.Emax = max(self.DataBase[0])

                self.chosing_DBType()

            except pg.DatabaseError:
                messagebox.showinfo("Error", "The database has not loaded \n \n Check your database input")

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
        Button(text="Quadtree DB", command=self.Quadtree,master=self.win_DBType).grid(row=0, column=2)
        self.win_DBType.mainloop()

    def XY2IJ(self,N,E):
        '''
        The function get N,E coordinate and return the i,j indexes in the DB matrix
        '''
        i = int(np.floor((N - self.Nmin) / self.a))
        if i < 0:
            i = 0
        if i > len(self.DBMatrix) - 1:
            i = len(self.DBMatrix) - 1
        j = int(np.floor((E - self.Emin) / self.b))
        if j < 0:
            j = 0
        if j > len(self.DBMatrix[0]) - 1:
            j = len(self.DBMatrix[0]) - 1
        return i, j

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
        self.ratio = simpledialog.askfloat("the ratio between the edges", "please input the ratio between the edges",parent=self.top)

        Ly=self.Nmax-self.Nmin
        Lx=self.Emax-self.Emin
        P=len(self.DataBase[0])/self.n0
        cellArea = (Lx*Ly)/P
        self.b=(cellArea/self.ratio)**0.5
        self.a=self.b*self.ratio
        Numx = int(np.ceil(Lx/self.b))
        Numy = int(np.ceil(Ly/self.a))
        self.DBMatrix = [[[] for i in range(Numx)] for j in range(Numy)] #creating a matrix with empty lists.
        for i in range(len(self.DataBase[0])): #loop for place the indexes of the coordinate in the relevant cell.
            E=self.DataBase[0][i]
            N=self.DataBase[1][i]
            ny,nx = self.XY2IJ(int(N),int(E))
            self.DBMatrix[ny][nx].append(i)

        self.win_DBType.quit()
        self.win_DBType.destroy()

        self.ploting()


    def Quadtree(self):
        '''
        Mark the database type as Quadtree and create the quadtree (by calling to the class) according to the maximum point in cell that it get from the user.
        ALso it close the small window and call to ploting function
        '''
        self.DataBaseType="Quadtree"
        self.nMAX = simpledialog.askinteger("maximum point in a cell", "please input the maximum points in one cell",parent=self.top)

        self.tree = quadtree(self.nMAX,self.DataBase,self.Emax,self.Nmax,self.Emin,self.Nmin,self.ax,self.canvas)

        self.win_DBType.quit()
        self.win_DBType.destroy()
        self.ploting()

    def ploting(self):
        '''
        Functin that printing the points, and print the database form acording to the type that selected by the user.
        :return:
        :rtype:
        '''
        if self.DataBaseType != None:
            Label(self.top, text="click on the scroller for selecting coordinates from plot",
                  font=('Times New Roman', 15)).grid(row=1, column=0, columnspan=10)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=10)
            self.ax.scatter(list(self.DataBase[0]), list(self.DataBase[1]), color='k', s=0.1)# plot all the points
            self.canvas.draw()
            self.canvas.mpl_connect('button_press_event', self.searchingCenter) # start the evet of click on the plot for choosing cocordinate.
            self.is_plot=True
            if self.show_toolbar:  # Display the toolbar
                self.toolbar_frame.grid(row=3, column=0, columnspan=9)
                self.navigation = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
                self.navigation.update()
                self.show_toolbar = False
            rect = patches.Rectangle((self.Emin,self.Nmin),self.Emax-self.Emin,self.Nmax-self.Nmin,edgecolor='red', facecolor='none', linewidth='0.5')
            self.ax.add_patch(rect) # plot the blocking Rectangle
            self.canvas.draw()

            if self.DataBaseType == "byArea":
                Ei = self.Emin
                while Ei <= self.Emax: # plot the vertical lines
                    self.ax.plot([Ei, Ei], [self.Nmin, self.Nmax], color='red', linewidth='0.5')
                    Ei += self.b
                Ni = self.Nmin
                while Ni <= self.Nmax: #plot the horizontal lines
                    self.ax.plot([self.Emin, self.Emax],[Ni, Ni],color='red', linewidth='0.5')
                    Ni += self.a
                self.canvas.draw()

            elif self.DataBaseType == 'Quadtree':
                self.tree.PlotBorder()

        else:
            messagebox.showinfo('Error', 'Please chose DB type')

    def Add_Coor_R(self):
        '''
        Function that ask for 3 parameters of serching: N,E and radius
        and plot the circle of the searching area
        '''
        if self.is_DB:
            if not self.is_selection:
                self.SearchE = simpledialog.askfloat("choose E coordinate", "E coordinate of the searching center", parent=self.top)
                self.SearchN = simpledialog.askfloat("choose N coordinate", "N coordinate of the searching center", parent=self.top)
                self.SearchR = simpledialog.askfloat("choose searching radius", "searching radius", parent=self.top)
                if self.Nmin <= self.SearchN <= self.Nmax and self.Emin <= self.SearchE <= self.Emax:
                    self.ax.scatter([self.SearchE],[self.SearchN],color = 'green', s=3)
                    self.canvas.draw()
                    circle = plt.Circle((self.SearchE, self.SearchN), self.SearchR, fill=None, color='green', linewidth=0.5)
                    self.ax.add_patch(circle)
                    self.canvas.draw()
                    self.is_selection = True
                    self.clicks = 1 #mark one click, for disable selecting another serching area befor searching.
                else:
                    messagebox.showinfo('Error', 'your point is out of range, please choose another one')
            else:
                messagebox.showinfo('Error', 'Please clear your last selection')
        else:
            messagebox.showinfo('Error', 'Please load DB file')

    def searchingCenter(self, event):
        '''
        Function that start when the user click on the roller, mark the point as center of searching area
        and ask for radius.
        At least it plot the circle of searching area.
        '''
        if event.button == 2:
            if not self.clicks:
                self.SearchE = event.xdata
                self.SearchN = event.ydata
                if self.Nmin <= self.SearchN <= self.Nmax and self.Emin <= self.SearchE <= self.Emax:
                    self.clicks = 1
                    self.SearchR = simpledialog.askfloat("choose searching radius", "searching radius", parent=self.top)
                    self.ax.scatter([self.SearchE],[self.SearchN],color = 'green', s=3)
                    self.canvas.draw()
                    circle = plt.Circle((self.SearchE, self.SearchN), self.SearchR, color='green',fill=None,linewidth=0.5)
                    self.ax.add_patch(circle)
                    self.canvas.draw()
                    self.is_selection = True
                else:
                    messagebox.showinfo('Error', 'your point is out of range, please choose another one')
            else:
                messagebox.showinfo('Error', 'Please clear your last selection')

    def Search(self):
        '''
        search the points in the searching area, un 3 different ways, according to the DB type
        at least it call for the function TableMaker and plot_Selection_Points
        '''
        if self.is_DB:
            start_time = time.time()
            if self.SearchE and self.SearchN and self.SearchR:
                p0 = Point2D(self.SearchE, self.SearchN)
                if self.DataBaseType =='Basic':
                    for i in range(len(self.DataBase[0])):
                        point1 = Point2D(float(self.DataBase[0][i]),float(self.DataBase[1][i]))
                        if point1.Square_distance_to_point(p0) <= self.SearchR**2:
                            self.selection_list.append((float(self.DataBase[0][i]),float(self.DataBase[1][i])))
                    end_time = time.time()
                    self.TableMaker()
                    self.plot_Selection_Points()
                elif self.DataBaseType == "byArea":
                    Nmin1 = self.SearchN - self.SearchR
                    Nmax1 = self.SearchN + self.SearchR
                    Emin1 = self.SearchE - self.SearchR
                    Emax1 = self.SearchE + self.SearchR
                    n_up_rightX,n_up_rightY = self.XY2IJ(Nmax1,Emax1)
                    n_down_leftX,n_down_leftY = self.XY2IJ(Nmin1,Emin1)
                    for i in range(n_down_leftX, n_up_rightX+1): #loop that check all of the points in the overlap cells with the searching area
                        for j in range(n_down_leftY, n_up_rightY+1):
                            for k in self.DBMatrix[i][j]:
                                if Point2D(float(self.DataBase[0][k]),float(self.DataBase[1][k])).Square_distance_to_point(p0) <= self.SearchR ** 2:
                                    self.selection_list.append((float(self.DataBase[0][k]),float(self.DataBase[1][k])))
                    end_time = time.time()
                    self.TableMaker()
                    self.plot_Selection_Points()
                else:  #for Quadtree
                    Nmin1 = self.SearchN - self.SearchR
                    Nmax1 = self.SearchN + self.SearchR
                    Emin1 = self.SearchE - self.SearchR
                    Emax1 = self.SearchE + self.SearchR
                    self.selection_list=self.tree.searchInTree(Nmin1,Nmax1,Emin1,Emax1,p0,self.SearchR,self.selection_list)
                    end_time = time.time()
                    self.TableMaker()
                    self.plot_Selection_Points()
                SearchingTime = round(end_time - start_time,3)
                messagebox.showinfo('Searching Time', f'Searching Time: {SearchingTime}',parent=self.win_PointsTable)
            else:
                messagebox.showinfo('Error', 'Please select selection area')

        else:
            messagebox.showinfo('Error', 'Please load DB file')



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
            self.SearchE = 0
            self.SearchN = 0
            self.SearchR = 0
            self.selection_list = []
            self.is_selection = False
            self.clicks = 0
            try:
                for i in self.selection_Table.get_children():
                    self.selection_Table.delete(i)
            finally:
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
            self.SearchE = 0
            self.SearchN = 0
            self.SearchR = 0
            self.selection_list = []
            self.is_selection = False
            self.clicks = 0
            try:
                for i in self.selection_Table.get_children():
                    self.selection_Table.delete(i)
            finally:
                return


        else:
            messagebox.showinfo('Error', 'Please load DB file')



    def TableMaker(self):
        '''
        Create new window and display there a table of all the selected points
        :return:
        :rtype:
        '''
        self.win_PointsTable = Tk()
        self.win_PointsTable.title('Selected points table')
        self.selection_Table = ttk.Treeview(self.win_PointsTable, columns=["N", "E"], show="headings")
        for i in self.selection_Table.get_children(): #delete the points of the last selection
            self.selection_Table.delete(i)
        for col in ["N", "E"]:
            self.selection_Table.heading(col, text=col)
            self.selection_Table.column(col, anchor="center")
        self.selection_Table.grid(row=1, column=10, columnspan=10)
        for P in self.selection_list:  #insert the new points
            self.selection_Table.insert("", "end", values=P)

    def plot_Selection_Points(self):
        '''
        Plot in a new color, the selected points
        '''
        listX = []
        listY = []
        for point in self.selection_list:
            listX.append(point[0])
            listY.append(point[1])
        self.ax.scatter(listX, listY, color='blue', s=2)
        self.canvas.draw()



    def Quit(self):  # Close the program
        self.top.quit()
        self.top.destroy()

