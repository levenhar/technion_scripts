
from tkinter import *
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import pandas as pd
from Point2D import Point2D
from math import inf
from Route import Route
from tkinter import messagebox





"""student_Name - Mor Levenhar and Tchelet Lev
Student_Id - 318301124, 206351611
Course Number - 014845
Course Name - introduction to computer mapping
Home Work number 4"""


class UI:
    def __init__(self, root):  # This class create a User Interface that shows the shortest route
        self.top = root
        self.top.wm_title("Shortest Route Calculation - H.W4")  # Define the title of the UI window
        self.ver = None  # Hold the vertices csv file
        self.edge = None  # Hold the edges csv file
        self.fig = Figure(figsize=(6, 6), dpi=100)  # Creating a figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)  # A tk.DrawingArea.
        self.toolbar_frame = Frame(self.top)  # Creating a toolbar
        self.navigation = None
        self.x_list = []  # Create a list with the x values
        self.y_list = []  # Create a list with the y values
        self.ax = self.fig.add_subplot(111)  # Create the axis
        self.ChooseStart = True  # Variable which help to check if the user pressed the selection button
        self.ChooseEnd = True  # Variable which help to check if the user pressed the selection button
        self.MarkStart = True  # Variable which help to limit the number of presses for the destination point
        self.MarkEnd = True  # Variable which help to limit the number of presses for the origin point
        self.start = None  # Holding the start point
        self.end = None  # Holding the end point
        self.route = None  # Holding the plot of the shortest route
        self.show_toolbar = True
        self.is_ver = False  # If the user uploaded vertices file
        self.is_edge = False  # If the user uploaded edges file

    def importVertices(self):  # Let the user upload a csv file and read it
        self.top.filename = filedialog.askopenfilename(initialdir="/", title="select vertices csvfile",
                                                       filetypes=(("csvfiles", " *.csv"),))
        self.ver = pd.read_csv(self.top.filename)
        self.is_ver = True

    def importEdges(self):  # Let the user upload a csv file and read it
        self.top.filename = filedialog.askopenfilename(initialdir="/", title="select edges csvfile",
                                                       filetypes=(("csvfiles", " *.csv"),))
        self.edge = pd.read_csv(self.top.filename)
        self.is_edge = True

    def Ploting(self):  # The function shows the points and edges on the canvas and display the toolbar
        pass
        if self.is_ver and self.is_edge:
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=9)
            self.x_list = self.ver["POINT_X"].values.tolist()  # Create a list with the x values
            self.y_list = self.ver["POINT_Y"].values.tolist()  # Create a list with the y values
            start_list = self.edge["POINT_0"].values.tolist()  # Create a list with the origin points
            end_list = self.edge["POINT_1"].values.tolist()  # Create a list with the destination points
            self.ax.scatter(self.x_list, self.y_list, color='k')  # Display the points on the canvas
            for i in range(len(start_list)):  # Display the edges on the canvas
                x_start = self.x_list[start_list[i]]
                y_start = self.y_list[start_list[i]]
                x_end = self.x_list[end_list[i]]
                y_end = self.y_list[end_list[i]]
                self.ax.plot([x_start, x_end], [y_start, y_end], color='k')
            if self.show_toolbar:  # Display the toolbar
                self.toolbar_frame.grid(row=2, column=0, columnspan=9)
                self.navigation = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
                self.navigation.update()
                self.show_toolbar = False
        else:
            messagebox.showinfo('Error','Please upload files')  # Error message in case that the user didn't upload a file

    def ClosestVertices(self, event):  # The function find and display the closest point and paint it
        x = event.xdata
        y = event.ydata
        first_d = FALSE
        first_o = FALSE
        txt = ""
        """Check that the user chose only one point"""
        if self.ChooseStart and not self.MarkStart:
            self.ChooseStart = FALSE
            first_o = True
        if self.ChooseEnd and not self.MarkEnd:
            self.ChooseEnd = FALSE
            first_d = True
        my_point = Point2D(x, y,"")
        min_dist = inf
        min_index = 0
        P_list = []
        for i in range(len(self.x_list)):  # Creating a list of the points from the vertices file
            P_list.append(Point2D(self.x_list[i], self.y_list[i],""))
        for i in range(len(self.x_list)):  # Find the closest point to the chosen one
            if my_point.distance_to_point(P_list[i]) < min_dist:
                min_dist = my_point.distance_to_point(P_list[i])
                min_index = i
        if first_o or (not first_o and first_d):
            if first_o:
                self.start = min_index
                txt = "start"
            if not first_o and first_d:
                self.end = min_index
                txt = "end"
            self.ax.scatter([self.x_list[min_index]], [self.y_list[min_index]], color='r', s=100,
                            edgecolors='k')
            self.ax.annotate(txt, (self.x_list[min_index],self.y_list[min_index]))  # Display the start, end text
            self.canvas.draw()
            first_o = FALSE

    def StartPoint(self):  # Let the user choose an origin point
        if self.is_ver and self.is_edge:
            self.canvas.mpl_connect('button_press_event', self.ClosestVertices)
            self.MarkStart = False
        else:
            messagebox.showinfo('Error', 'Please upload files')  # Error message if the user didn't import files

    def EndPoint(self):  # Let the user choose a destination point
        if self.is_ver and self.is_edge:
            self.canvas.mpl_connect('button_press_event', self.ClosestVertices)
            self.MarkEnd = False
        else:
            messagebox.showinfo('Error', 'Please import files')  # Error message if the user didn't import files

    def calculate_Route(self):  # Calculate and display the shortest route
        if self.is_ver and self.is_edge:
            if not self.MarkStart and not self.MarkEnd:
                route = Route(self.ver, self.edge, self.start, self.end)
                shortest_route = route.dijkstra()  # Call to the dijkstra function from Route class
                for i in range(len(shortest_route) - 1):  # Dispaly the shortest route
                    self.route = self.ax.plot([self.x_list[shortest_route[i]], self.x_list[shortest_route[i + 1]]],
                                              [self.y_list[shortest_route[i]], self.y_list[shortest_route[i + 1]]],
                                              color='r', linewidth=3)
                self.canvas.draw()
            else:
                messagebox.showinfo('Error',
                                    'Please select start and end points')  # Error message if the user didn't chose the point
        else:
            messagebox.showinfo('Error', 'Please import csv files')  # Error message if the user didn't import files

    def clear(self):  # Clear the canvas and display only the map
        self.ax.clear()
        self.Ploting()
        self.canvas.draw()
        self.ChooseStart = True
        self.MarkStart = True
        self.ChooseEnd = True
        self.MarkEnd = True

    def Quit(self):  # Close the program
        self.top.quit()
        self.top.destroy()

    def ui(self):  # Place the Buttons on the screen
        Button(text="Import Vertices csv", command=self.importVertices).grid(row=0,column=0)
        Button(text="Import Edges csv", command=self.importEdges).grid(row=0,column=1)
        Button(text="Show the plot", command=self.Ploting).grid(row=0, column=2)
        Button(text="Select start point", command=self.StartPoint,).grid(row=0, column=3)
        Button(text="Select end point", command=self.EndPoint).grid(row=0,column=4)
        Button(text="Calculate Route", command=self.calculate_Route).grid(row=0,column=5)
        Button(text="Clear", command=self.clear).grid(row=0, column=6)
        Button(text="Quit", command=self.Quit).grid(row=0, column=7)