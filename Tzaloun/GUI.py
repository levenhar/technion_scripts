import math
from tkinter import *
from tkinter import messagebox
import time
import csv
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from tkinter import simpledialog
import numpy as np
from Point2D22 import Point2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import dill


class UI:
    def __init__(self,root):
        self.top = root  #main window of GUI
        #self.win_DBType= None  #secondry window for choching DB type
        self.win_Tzluon_Table = None #secondry window for display the selected points
        self.Tzluon_Table = "" #pd.dataframe that contain the data
        self.Pratim_Table=""
        self.is_control = False
        self.controlPoints=[]
        self.is_table = False
        self.azimuth=[]
        self.PratimPoints=[["Name","X","Y","H","Code","Sxx","Syy","Sxy"]]
        #self.DataBaseType = None # mark the DB type that chosen
        self.fig = Figure(figsize=(6, 6), dpi=100, )  # Creating a figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)  # A tk.DrawingArea.
        self.toolbar_frame = Frame(self.top)  # Creating a toolbar
        self.ax = self.fig.add_subplot(111)  # Create the axis
        self.ax.axis('equal')
        self.show_toolbar = True #for creating the toolbar only in the first time.
        self.is_plot = False # mark if there is a plot all ready
        self.is_Adjast = False
        self.Lb=np.array([])
        self.B=np.array([])
        self.w=np.array([])
        self.P=np.array([])
        self.TS_Azimuths=None
        self.Slb=np.array([])
        self.V = np.array([])
        self.M = np.array([])
        self.Points=[]
        self.Names=[]
        self.Sla=np.array([])
        self.C=np.array([])
        self.Sx=np.array([])
        self.La =np.array([])
        self.Sigapost=None
        self.VTPV=None
        #self.selection_list = [] #list of all the selected points
        #self.selection_Table = ttk.Treeview() #the object that cuntain the seleced point table


    def ui(self):
        # Place the Buttons on the screen
        Button(text="Import Control Points", command=self.ControlPoints,master=self.top).grid(row=0, column=0)
        Button(text="Import Tzloun Table", command=self.TzlounTable, master=self.top).grid(row=0, column=1)
        Button(text="Adjust", command=self.Adjust, master=self.top).grid(row=0, column=2)
        Button(text="Add Heights", command=self.AddHeight, master=self.top).grid(row=0, column=3)
        Button(text="Pratim", command=self.Pratim, master=self.top).grid(row=0, column=4)
        Button(text="Ratz Nitzav", command=self.Pratim_Ratz_Nitzav, master=self.top).grid(row=0, column=5)
        Button(text="Load Adjusted Tzloun", command=self.Load, master=self.top).grid(row=0, column=6)
        Button(text="Quit", command=self.Quit,master=self.top).grid(row=0, column=7)


    def ControlPoints(self):
        self.is_control = False
        self.top.filename = filedialog.askopenfilename(initialdir="/", title="select Control Points file.",
                                                       filetypes=(("csvfiles", " *.csv"),))
        ControlP = pd.read_csv(self.top.filename)
        try:
            for i in range(len(ControlP.iloc[:,0])):
                self.controlPoints.append(Point2D(ControlP.at[i,'X'],ControlP.at[i,'Y'],ControlP.at[i,'Name']))
            self.is_control=True
        except KeyError:
            messagebox.showinfo('Error', 'Make sure the headers is X, Y and Name')

        if self.is_control:
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=10)
            self.ax.grid()
            for P in self.controlPoints:
                self.ax.scatter([P.my_y], [P.my_x], color="red",marker="^")
                self.ax.annotate(str(P.my_id), (P.my_y + 5, P.my_x + 5))
                self.canvas.draw()


            if self.show_toolbar:  # Display the toolbar
                self.toolbar_frame.grid(row=3, column=0, columnspan=9)
                self.navigation = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
                self.navigation.update()
                self.show_toolbar = False

    def AddHeight (self):
        for P in self.Points:
            P.my_h=simpledialog.askfloat("insert height",f"insert the height of Point: {P.my_id}",parent=self.top)

        for P in self.Points:
            P.Sh=simpledialog.askfloat("insert height accuracy",f"insert the height accuracy of Point: {P.my_id}",parent=self.top)

        Pointslist = [["Name","X", "Y","H",'Height Accuracy']]
        for P in self.Points:
            Pointslist.append([P.my_id,P.my_x, P.my_y,P.my_h,P.Sh])
        PointsArray = np.array(Pointslist)
        np.savetxt(f"Xa.csv", PointsArray, delimiter=",", fmt='%s')


    def TzlounTable(self):
        if not self.is_control:
            messagebox.showinfo('Error', 'import the control points first')
        else:
            self.top.filename = filedialog.askopenfilename(initialdir="/", title="select Tzloun table file",filetypes=(("csvfiles", " *.csv"),))
            self.Tzluon_Table = pd.read_csv(self.top.filename)
            self.is_table = True

            L=self.Tzluon_Table.iloc[1:-1,6]
            h=self.Tzluon_Table.iloc[1:-1, 10]
            a=self.Tzluon_Table.iloc[:,5]
            L1=self.Tzluon_Table.iloc[1:-1,15]
            h1 =self.Tzluon_Table.iloc[1:-1, 19]
            a1=self.Tzluon_Table.iloc[:,14]
            self.Names=self.Tzluon_Table.iloc[:,0]


            self.Lm=[]
            for i in range(0,len(L)-1,2):
                pass
                self.Lm.append([(float(L.iat[i])*math.sin(math.pi/180*float(h.iat[i]))+float(L.iat[i+1])*math.sin(math.pi/180*float(h.iat[i+1]))+float(L1.iat[i])*math.sin(math.pi/180*(float(360-h1.iat[i])))+float(L1.iat[i+1])*math.sin(math.pi/180*(float(360-h1.iat[i+1]))))/4])

            am=[]
            for i in range(0,len(a),1):
                if a.iat[i]>a1.iat[i]:
                    am.append((a.iat[i]+a1.iat[i]+180)/2)
                else:
                    am.append((a.iat[i] + a1.iat[i] - 180) / 2)
            self.aa=[]
            for i in range(0,len(am)-1,2):
                ai=am[i+1]-am[i]
                while ai<0:
                    ai+=360
                self.aa.append([ai])

            self.Lb=np.array(self.Lm+self.aa)
            np.savetxt("Lb.csv", self.Lb, delimiter=",")


            self.azimuth = [self.controlPoints[0].azimuth(self.controlPoints[1])]
            for i in range(len(self.aa)):
                azi = self.azimuth[i] + self.aa[i][0] - 180
                while azi > 360:
                    azi -= 360
                while azi < 0:
                    azi += 360
                self.azimuth.append(azi)


            n=len(self.aa)
            LLL=0
            for l1 in self.Lm:
                LLL+=l1[0]

            Waa=45*np.sqrt(n)/3600
            WL=1/3000

            w1 = self.azimuth[-1] - self.controlPoints[2].azimuth(self.controlPoints[3])
            if abs(w1)>Waa:
                messagebox.showinfo('Error', 'The angle error is too big')
                return
            else:
                www=-1
                if w1<0:
                    w1=-w1
                    www=1
                nn=round(w1*3600,0)//n
                nnn=round(w1*3600,0)%n
                nn=www*nn
                nnn=www*nnn
                ii=0
                azi_mekorav=self.azimuth[1:]
                for i in range(len(azi_mekorav)):
                    azi_mekorav[i]=azi_mekorav[i]+nn*(i+1)/3600+ii/3600
                    if nnn!=0:
                        azi_mekorav[i]+=(nnn/abs(nnn))/3600
                        if nnn>0:
                            nnn-=1
                            ii+=1
                        else:
                            nnn+=1
                            ii -= 1

                e3=(self.controlPoints[2].azimuth(self.controlPoints[3])-azi_mekorav[-1])*3600
                dx = []
                dy = []
                for i in range(len(self.Lm)):
                    dx.append(self.Lm[i][0] * math.cos(math.radians(azi_mekorav[i])))
                    dy.append(self.Lm[i][0] * math.sin(math.radians(azi_mekorav[i])))

                dxOld=dx[:]
                dyOld=dy[:]

                w2 = self.controlPoints[1].my_x + sum(dx) - self.controlPoints[2].my_x
                w3 = self.controlPoints[1].my_y + sum(dy) - self.controlPoints[2].my_y
                if (np.sqrt(w2**2+w3**2)/LLL)>WL:
                    messagebox.showinfo('Error', 'The distance error is too big')
                    return
                else:
                    for i in range(len(self.Lm)):
                        dx[i]+=-w2*self.Lm[i][0]/LLL
                        dy[i]+= -w3 * self.Lm[i][0] / LLL

                    X=[self.controlPoints[1].my_x]
                    Y=[self.controlPoints[1].my_y]
                    for i in range(len(self.Lm)):
                        X.append(X[-1]+dx[i])
                        Y.append(Y[-1]+dy[i])
                    dx.append("")
                    dy.append("")

                    Table=[["Name"]+list(self.Names[::2]),["Azimuth [°]"]+self.azimuth[1:],["dx [m]"]+dxOld+[""],["dy [m]"]+dyOld+[""],["Fixed Azimuths[°]"]+azi_mekorav,["Fixed dx [m]"]+dx,["Fixed dy [m]"]+dy,["Fixed X [m]"]+X,["Fixed Y[m]"]+Y]
                    Table=np.transpose(np.array(Table))
                    np.savetxt("semi_adjust.csv", Table, delimiter=",",fmt='%s')



                    e1=self.controlPoints[2].my_x-X[-1]
                    e2=self.controlPoints[2].my_y-Y[-1]

                    with open('Tzloun_Mekorav.txt', 'w') as f:
                        f.write(f'angele error[\'\']={round(w1*3600,2)} \nX error[m]={round(w2,5)} \nY error[m]={round(w3,5)} \nAfter the Adjusment Mekurav: \nangele error[\'\']={round(e3,2)} \nX error[m]={round(e1,5)} \nY error={round(e2,5)}')
                    messagebox.showinfo('successful upload', 'The table upload was successful')
    def Adjust (self):
        if self.is_table and self.is_control:
            sa=simpledialog.askfloat("import the accuracy of the angles", "accuracy of the angles [\'\']", parent=self.top)/3600
            sl_m=simpledialog.askfloat("import the accuracy of the distances", "accuracy of the distances", parent=self.top)
            sl_ppm=simpledialog.askfloat("import the accuracy of the distances", "accuracy of the distances (ppm)", parent=self.top)
            s0 = simpledialog.askfloat("import the sigma apriori", "sigma apriori",parent=self.top)
            if sa ==0 and sl_m==0 and sl_ppm==0:
                self.P=np.eye(len(self.Lb))
                self.Slb = np.eye(len(self.Lb))
            else:
                self.Slb = np.zeros((len(self.Lb), len(self.Lb)))
                for i in range(len(self.Lm)):
                    self.Slb[i][i] = (sl_m ** 2 + (self.Lm[i][0] * (sl_ppm / (10 ** 6))) ** 2)
                for i in range(len(self.Lm), len(self.Lm) + len(self.aa)):
                    self.Slb[i][i] = (sa) ** 2
                self.P = s0 * np.linalg.inv(self.Slb)

            dx = []
            dy = []
            for i in range(len(self.Lm)):
                dx.append(self.Lm[i][0] * math.cos(math.radians(self.azimuth[i + 1])))
                dy.append(self.Lm[i][0] * math.sin(math.radians(self.azimuth[i + 1])))

            w1 = self.azimuth[-1] - self.controlPoints[2].azimuth(self.controlPoints[3])
            w2 = self.controlPoints[1].my_x + sum(dx) - self.controlPoints[2].my_x
            w3 = self.controlPoints[1].my_y + sum(dy) - self.controlPoints[2].my_y
            self.w = np.array([[w1], [w2], [w3]])
            np.savetxt("W0.csv", self.w, delimiter=",")

            n=0
            while self.w[0, :] > 1*10^-5 and max(self.w[1:2, :]) > 0.00004:
                b1=[]
                for i in range(len(self.Lm)):
                    b1.append(0)
                for i in range(len(self.aa)):
                    b1.append(1)

                b2=[]
                b3 = []
                for i in range(len(self.Lm)):
                    b2.append(math.cos(math.radians(self.azimuth[i+1])))
                    b3.append(math.sin(math.radians(self.azimuth[i + 1])))
                for i in range(len(self.aa)-1):
                    b2.append(-sum(dy[i:])/206265*3600)
                    b3.append(sum(dx[i:])/206265*3600)
                b2.append(0)
                b3.append(0)

                self.B=np.array([b1,b2,b3])
                np.savetxt(f"B{n}.csv", self.B, delimiter=",")
                self.M=np.matmul(np.matmul(self.B,self.Slb/s0),np.transpose(self.B))
                np.savetxt(f"M{n}.csv", self.M, delimiter=",")
                self.V=-np.matmul(np.matmul(np.matmul(self.Slb/s0,np.transpose(self.B)),np.linalg.inv(self.M)),self.w)
                np.savetxt(f"V{n}.csv", self.V, delimiter=",")
                self.La=self.Lb+self.V
                np.savetxt(f"La{n+1}.csv", self.La, delimiter=",")

                self.aa=self.La[len(self.Lm):, :]
                self.Lm=self.La[:len(self.Lm), :]


                self.azimuth = [self.controlPoints[0].azimuth(self.controlPoints[1])]
                for i in range(len(self.aa)):
                    azi = self.azimuth[i] + self.aa[i,0] - 180
                    while azi > 360:
                        azi -= 360
                    while azi < 0:
                        azi += 360
                    self.azimuth.append(azi)

                dx = []
                dy = []
                for i in range(len(self.Lm)):
                    dx.append(self.Lm[i,0] * math.cos(math.radians(self.azimuth[i + 1])))
                    dy.append(self.Lm[i,0] * math.sin(math.radians(self.azimuth[i + 1])))

                w1 = self.azimuth[-1] - self.controlPoints[2].azimuth(self.controlPoints[3])
                w2 = self.controlPoints[1].my_x + sum(dx) - self.controlPoints[2].my_x
                w3 = self.controlPoints[1].my_y + sum(dy) - self.controlPoints[2].my_y
                self.w = np.array([[w1], [w2], [w3]])
                n+=1
                np.savetxt(f"W{n}.csv", self.w, delimiter=",")

            with open('All_azimuths.csv', 'w') as f:
                write = csv.writer(f)
                write.writerow(self.azimuth)

            self.Points.append(Point2D(self.controlPoints[1].my_x + dx[0],self.controlPoints[1].my_y + dy[0],self.Names[2]))
            self.Points[0].plot(self.ax)
            for i in range(len(dx)-2):
                self.Points.append(Point2D(self.Points[-1].my_x+dx[i+1],self.Points[-1].my_y+dy[i+1],self.Names[2*(i+2)]))
                self.Points[i+1].plot(self.ax)
                self.canvas.draw()


            Pointslist=[["X","Y","Name"]]
            for P in self.Points:
                Pointslist.append([P.my_x,P.my_y,P.my_id])
            PointsArray=np.array(Pointslist)
            np.savetxt(f"Xa.csv", PointsArray, delimiter=",",fmt='%s')

            self.C = np.zeros((len(PointsArray)*2-2,len(self.Lb)))
            for i in range(0,(len(PointsArray)-1)*2-1,2):
                for j in range(i//2+1):
                    self.C[i,j]=math.cos(math.radians(self.azimuth[j+1]))
                    self.C[i+1,j] = math.sin(math.radians(self.azimuth[j + 1]))
                    self.C[i,j+len(self.Lm)]=-sum(dy[j:i//2+1])/(206265/3600)
                    self.C[i+1,j+len(self.Lm)]=sum(dx[j:i//2+1])/(206265/3600)

            np.savetxt(f"C_TSpoints.csv", self.C, delimiter=",", fmt='%s')

            self.Sigapost=np.matmul(np.matmul(np.transpose(self.V),self.P),self.V)/3
            self.Sla=self.Sigapost*(np.linalg.inv(self.P)-np.matmul(np.matmul(np.matmul(np.matmul(np.linalg.inv(self.P),np.transpose(self.B)),np.linalg.inv(self.M)),self.B),np.linalg.inv(self.P)))
            np.savetxt(f"Sla.csv", self.Sla, delimiter=",", fmt='%s')


            self.Sx=np.matmul(np.matmul(self.C,self.Sla),np.transpose(self.C))
            np.savetxt(f"Sx.csv", self.Sx, delimiter=",", fmt='%s')

            for i in range(0,2*len(self.Points)-1,2):
                self.Points[i//2].Sxx=self.Sx[i,i]
                self.Points[i//2].Syy=self.Sx[i+1,i+1]
                self.Points[i//2].Sxy=self.Sx[i,i+1]
                self.Points[i//2].ErrorElipse()

                #in the angel there is a (-) because it defied clockwise
                #the ellipse scale is X500 from the chart scale

                ellipse = Ellipse(xy=(self.Points[i//2].my_y, self.Points[i//2].my_x), width=self.Points[i//2].Smin*500, height=self.Points[i//2].Smax*500, edgecolor='r',angle=-self.Points[i//2].phi*180/np.pi, fc='None', lw=1)
                self.ax.add_patch(ellipse)

            ElliList = [['Name','Sxx','Syy','Sxy','Smax', 'Smin','Smax - 95%', 'Smin- 95%', 'phi']]
            for P in self.Points:
                ElliList.append([P.my_id,P.Sxx,P.Syy,P.Sxy,P.Smax, P.Smin,P.Smax*2.447, P.Smin*2.447, P.phi*180/np.pi])
            ElliArray = np.array(ElliList)
            np.savetxt(f"Error_Ellipses.csv", ElliArray, delimiter=",", fmt='%s')


            messagebox.showinfo('successful Adjusment', f'The Ajusment Prosses was successful\n (vTpv)^2={round(self.Sigapost[0][0],6)}')


            with open('Tzloun_Accuracy.txt', 'w') as f:
                f.write(f'SigmaApriori={s0} \nSigmaPost[-]={round(self.Sigapost[0][0],15)} \nS_angel={sa*3600} \nS_distance[m]={sl_m} \nS_distance[ppm]={sl_ppm}')

        else:
            messagebox.showinfo('Error', 'Please load Control points and Tzloun Table')

    def AzimuthsCalc(self):
        fromm = ["From", self.controlPoints[1].my_id]
        to = ["To", self.Points[0].my_id]
        for i in range(len(self.Points) - 1):
            fromm.append(self.Points[i].my_id)
            to.append(self.Points[i + 1].my_id)
        fromm.append(self.Points[-1].my_id)
        to.append(self.controlPoints[2].my_id)

        C = np.zeros((len(self.Points) + 1, len(self.Points) + 1))
        for i in range(len(self.Points) + 1):
            for j in range(i + 1):
                C[i][j] = 1

        S = self.Sla[len(self.Points) + 1:-1, len(self.Points) + 1:-1]
        Sazi = np.matmul(np.matmul(C, S), np.transpose(C))

        Sazimuth = ["S_azimuth [°^2]"]
        for i in range(len(Sazi)):
            Sazimuth.append(Sazi[i][i])

        azimuth = np.transpose(np.array([fromm, to, ["azimuth [°]"] + self.azimuth[1:-1], Sazimuth]))
        self.TS_Azimuths = azimuth[1:]
        np.savetxt(f"azimuths.csv", azimuth, delimiter=",", fmt='%s')

    def Pratim(self):
        self.top.filename = filedialog.askopenfilename(initialdir="/", title="select the Pritim file",
                                                       filetypes=(("csvfiles", " *.csv"),))
        self.Pratim_Table = pd.read_csv(self.top.filename)
        try:
            if self.TS_Azimuths==None:
                self.AzimuthsCalc()
        except:
            pass

        P_TS = None

        sl=(simpledialog.askfloat("import the accuracy of distance", "accuracy of the distance [m]", parent=self.top))**2
        sh =(simpledialog.askfloat("import the accuracy of the vertical angles", "accuracy of the vertical angles [\'\']", parent=self.top)/206265)**2
        sB =(simpledialog.askfloat("import the accuracy of the Horizontal Direction", "accuracy of the Horizontal Direction [\'\']", parent=self.top)/206265)**2
        sb=(simpledialog.askfloat("import the accuracy of b", "accuracy of b [m]", parent=self.top))**2
        sa=(simpledialog.askfloat("import the accuracy of a", "accuracy of a [ppm]", parent=self.top))**2
        sc=(simpledialog.askfloat("import the accuracy of c", "accuracy of c [\'\']", parent=self.top)/206265)**2
        sepsi =(simpledialog.askfloat("import the accuracy of epsilon", "accuracy of epsilon [\'\']", parent=self.top)/206265)**2
        sii=(simpledialog.askfloat("import the accuracy of i", "accuracy of i [\'\']", parent=self.top)/206265)**2
        Sts = (simpledialog.askfloat("import the accuracy of TS", "accuracy of TS height [m]", parent=self.top)) ** 2
        sp = (simpledialog.askfloat("import the accuracy of Prism", "accuracy of Prism Height [m]", parent=self.top)) ** 2

        with open('Accuracy_pratim.txt', 'w') as f:
            f.write(
                f'S_distance[m]={np.sqrt(sl)} \nS_vertical angles[\'\']={np.sqrt(sh)*206265} \nS_Horizontal Direction[\'\']={np.sqrt(sB) * 206265} \nS_b[m]={np.sqrt(sb)} \nS_a[ppm]={np.sqrt(sa)} \nS_i [\'\']={np.sqrt(sii)*206265} \nS_c [\'\']={np.sqrt(sc)*206265} \nS_epsi [\'\']={np.sqrt(sepsi)*206265} \nS_TS height [m]={np.sqrt(Sts)} \nS_Prism height [m]={np.sqrt(sp)}')


        Points=[["Name","From","X [m]","Y [m]","H [m]","Sxx [m^2]","Syy [m^2]","Sxy [m^2]","Smax [m]","Smin [m]","Phi [°]","Sh [m]"]]

        for i in range(len(self.Pratim_Table.iloc[:, 0])):
            for P in self.Points:
                if self.Pratim_Table.at[i, 'From'] == P:
                    P_TS=P
                    break
            for j in range(len(self.TS_Azimuths)):
                if P==str(self.TS_Azimuths[j][0])and self.Pratim_Table.at[i, 'Ref']==self.TS_Azimuths[j][1]:
                    self.Pratim_Table.at[i, 'Azimuth']=self.TS_Azimuths[j][2]
                    self.Pratim_Table.at[i, 'S_Azimuth'] = self.TS_Azimuths[j][3]
                    break
                elif P==str(self.TS_Azimuths[j][1]) and self.Pratim_Table.at[i, 'Ref']==self.TS_Azimuths[j][0]:
                    self.Pratim_Table.at[i, 'Azimuth'] = float(self.TS_Azimuths[j][2]) +180
                    while self.Pratim_Table.at[i, 'Azimuth'] > 360:
                        self.Pratim_Table.at[i, 'Azimuth'] = self.Pratim_Table.at[i, 'Azimuth'] - 360
                    self.Pratim_Table.at[i, 'S_Azimuth'] = self.TS_Azimuths[j][3]
                    break
            a=self.Pratim_Table.at[i, 'a']
            b=self.Pratim_Table.at[i, 'b']
            self.Pratim_Table.at[i, 'Vertical Angel Correct']=self.Pratim_Table.at[self.Pratim_Table.at[i, 'i'], 'Vertical Angel']+self.Pratim_Table.at[i, 'epsi']/3600
            self.Pratim_Table.at[i, 'epsi_tag']=self.Pratim_Table.at[i, 'c']/np.cos(i/206265)+self.Pratim_Table.at[i, 'i']*np.tan(np.radians(self.Pratim_Table.at[i, 'Vertical Angel']))
            self.Pratim_Table.at[i, 'Horizontal Direction Correct']=self.Pratim_Table.at[i,"Horizontal Direction"]+self.Pratim_Table.at[i,"epsi_tag"]/3600
            self.Pratim_Table.at[i, 'Slope Distance Correct']=self.Pratim_Table.at[i, 'Slope Distance']*(1+(a*(10**-6)))+b
            self.Pratim_Table.at[i, 'Vertical Distance']=self.Pratim_Table.at[i, 'Slope Distance Correct']*np.sin(np.radians(self.Pratim_Table.at[i, 'Vertical Angel Correct']))
            self.Pratim_Table.at[i, 'Horizontal Distance'] =self.Pratim_Table.at[i, 'Slope Distance Correct']*np.cos(np.radians(self.Pratim_Table.at[i, 'Vertical Angel Correct']))
            self.Pratim_Table.at[i, 'Azimuth to Point'] = self.Pratim_Table.at[i, 'Azimuth'] -(360-self.Pratim_Table.at[i, 'Horizontal Direction Correct'])
            while self.Pratim_Table.at[i, 'Azimuth to Point'] > 360:
                self.Pratim_Table.at[i, 'Azimuth to Point'] = self.Pratim_Table.at[i, 'Azimuth to Point']-360
            while self.Pratim_Table.at[i, 'Azimuth to Point'] < 0:
                self.Pratim_Table.at[i, 'Azimuth to Point'] = self.Pratim_Table.at[i, 'Azimuth to Point'] + 360
            self.Pratim_Table.at[i, 'H']=P_TS.my_h+self.Pratim_Table.at[i, 'TS height']+self.Pratim_Table.at[i, 'Vertical Distance']-self.Pratim_Table.at[i, 'Prism height']
            self.Pratim_Table.at[i, 'X']=P_TS.my_x+self.Pratim_Table.at[i, 'Horizontal Distance']*np.cos(np.radians(self.Pratim_Table.at[i, 'Azimuth to Point']))
            self.Pratim_Table.at[i, 'Y'] = P_TS.my_y + self.Pratim_Table.at[i, 'Horizontal Distance'] * np.sin(np.radians(self.Pratim_Table.at[i, 'Azimuth to Point']))



            Cosaz=np.cos(np.radians(self.Pratim_Table.at[i, 'Azimuth to Point']))
            Sinaz=np.sin(np.radians(self.Pratim_Table.at[i, 'Azimuth to Point']))
            CosH=np.cos(np.radians(self.Pratim_Table.at[i, 'Vertical Angel Correct']))
            SinH=np.sin(np.radians(self.Pratim_Table.at[i, 'Vertical Angel Correct']))


            #error ellipse (x,y)
            CC=np.zeros((2,11))
            CC[0][0]=1
            CC[1][1]=1
            CC[0][2]=(1+a*10**-6)*Cosaz
            CC[1][2]=(1+a*10**-6)*Sinaz
            CC[0][3] = Cosaz*self.Pratim_Table.at[i, 'Slope Distance']*CosH/1000000
            CC[1][3] = Sinaz*self.Pratim_Table.at[i, 'Slope Distance']*CosH/1000000
            CC[0][4]=Cosaz
            CC[1][4]=Sinaz
            CC[0][5]= CC[0][6]=CC[0][7]= CC[0][8]= CC[0][9]= CC[0][10]= -Sinaz*self.Pratim_Table.at[i, 'Horizontal Distance']
            CC[1][5]= CC[1][6]= CC[1][7]= CC[1][8]= CC[1][9]=CC[1][10]= Cosaz*self.Pratim_Table.at[i, 'Horizontal Distance']
            CC[0][7] = CC[0][7]/CosH
            CC[1][7] = CC[1][7]/CosH
            CC[0][8] = CC[0][8] *((self.Pratim_Table.at[i, 'c']/206265)*SinH+(self.Pratim_Table.at[i, 'i']/206265))/CosH**2
            CC[1][8] = CC[1][8] *((self.Pratim_Table.at[i, 'c']/206265)*SinH+(self.Pratim_Table.at[i, 'i']/206265))/CosH**2
            CC[0][9] = CC[0][9] *-((self.Pratim_Table.at[i, 'c']/206265)*SinH+(self.Pratim_Table.at[i, 'i']/206265))/CosH**2
            CC[1][9] = CC[1][9] *-((self.Pratim_Table.at[i, 'c']/206265)*SinH+(self.Pratim_Table.at[i, 'i']/206265))/CosH**2
            CC[0][10] = CC[0][10] *SinH/CosH
            CC[1][10] = CC[1][10] *SinH/CosH


            Sigmai=[P_TS.Sxx,P_TS.Syy,sl,sa,sb,self.Pratim_Table.at[i, 'S_Azimuth']/180*np.pi,sB,sc,sh,sepsi,sii]

            SSS=np.diag(Sigmai)
            SSS[0][1]=SSS[1][0]=P_TS.Sxy

            Sp=np.matmul(np.matmul(CC, SSS), np.transpose(CC))

            PPP=Point2D(self.Pratim_Table.at[i, 'X'],self.Pratim_Table.at[i, 'Y'],self.Pratim_Table.at[i, 'To point'],self.Pratim_Table.at[i, 'H'],Sxx=Sp[0][0],Syy=Sp[1][1],Sxy=Sp[0][1])
            PPP.plot(self.ax,"Prat")
            self.PratimPoints.append(PPP)
            PPP.ErrorElipse()

            #error in H
            CC=np.array([1,1,(1+a*10**-6)*SinH,10**-6*SinH,SinH,CosH,-CosH,-1])
            SSS = np.diag([P_TS.Sh,Sts,sl,sa,sb,sh,sepsi,sp])
            Shh = np.matmul(np.matmul(CC, SSS), np.transpose(CC))
            PPP.Sh=np.sqrt(Shh)

            Points.append([PPP.my_id,P_TS.my_id,PPP.my_x,PPP.my_y,PPP.my_h,PPP.Sxx,PPP.Syy,PPP.Sxy,PPP.Smax,PPP.Smin,PPP.phi*180/np.pi,PPP.Sh])
        self.canvas.draw()
        self.Pratim_Table.to_csv('Pratim_Table.csv')
        np.savetxt(f"Pratim_with_errors.csv", Points, delimiter=",", fmt='%s')

    def Pratim_Ratz_Nitzav(self):
        self.top.filename = filedialog.askopenfilename(initialdir="/", title="select the Pritim file",
                                                       filetypes=(("csvfiles", " *.csv"),))
        self.Meshiha_Table = pd.read_csv(self.top.filename)

        try:
            if self.TS_Azimuths == None:
                self.AzimuthsCalc()
        except:
            pass

        P0=None
        for P in self.Points:
            if str(self.Meshiha_Table.at[0, 'From'])== P:
                P0 = P
                break

        for j in range(len(self.TS_Azimuths)):
            if P0==str(self.TS_Azimuths[j][0])and str(self.Meshiha_Table.at[0, 'Ref'])==self.TS_Azimuths[j][1]:
                aa=float(self.TS_Azimuths[j][2])
                Saa = float(self.TS_Azimuths[j][3])
                for P in self.Points:
                    if str(self.Meshiha_Table.at[0, 'Ref']) == P:
                        self.ax.plot([P0.my_y,P.my_y],[P0.my_x,P.my_x], linewidth=0.5)
                        self.canvas.draw()
                break
            elif P0==str(self.TS_Azimuths[j][1]) and self.Meshiha_Table.at[0, 'Ref']==self.TS_Azimuths[j][0]:
                aa = float(self.TS_Azimuths[j][2]) +180
                while aa > 360:
                    aa = aa - 360
                Saa = float(self.TS_Azimuths[j][3])
                for P in self.Points:
                    if str(self.Meshiha_Table.at[0, 'Ref']) == P:
                        self.ax.plot([P0.my_y,P.my_y],[P0.my_x,P.my_x], linewidth=0.5)
                        self.canvas.draw()
                break
        delta=(simpledialog.askfloat("import the accuracy of 90°", "accuracy of 90° [\'\']", parent=self.top)/206265)
        Sa = (simpledialog.askfloat("import the accuracy of a", "accuracy a [m]",parent=self.top))
        Sb = (simpledialog.askfloat("import the accuracy of b", "accuracy b [m]",parent=self.top))

        with open('Accuracy_Meshiha.txt', 'w') as f:
            f.write(
                f'S_90°[\'\']={delta*206265} \nS_a[m]={Sa} \nS_b[m]={Sb}')

        sinD=np.sin(delta)
        cosD=np.cos(delta)
        sina=np.sin(aa*np.pi/180)
        cosa=np.cos(aa*np.pi/180)
        X0=np.array([[P0.my_x],[P0.my_y]])
        R=np.array([[cosa,-sina],[sina, cosa]])
        C=np.zeros((2,5))
        C[0][0]=C[1][1]=1
        C[0][2]=cosa
        C[1][2]=sina
        C[0][3]=sinD*cosa-cosD*sina
        C[1][3]=sinD*sina+cosD*cosa
        SSS=np.diag([P0.Sxx,P0.Syy,Sa**2,Sb**2,np.radians(np.sqrt(Saa))**2])
        SSS[0][1]=SSS[1][0]=P0.Sxy
        PointCSV=[["Name","Code","X [m]","Y [m]","Sxx [m^2]","Syy [m^2]","Sxy [m^2]","Smax [m]","Smin [m]","Phi [°]"]]
        for i in range(1,len(self.Meshiha_Table.iloc[1:,2])+1):
            a=self.Meshiha_Table.at[i,"a"]
            b=self.Meshiha_Table.at[i,"b"]
            AB=np.array([[a+b*sinD],[b*sinD]])
            P=X0+np.matmul(R,AB)
            PPP=Point2D(P[0][0],P[1][0],self.Meshiha_Table.at[i,"To point"])
            self.PratimPoints.append(PPP)
            C[0][4]=-(a+b*sinD)*sina-(b*cosD)*cosa
            C[1][4]=(a+b*sinD)*cosa-(b*cosD)*sina
            SP=np.matmul(np.matmul(C,SSS),np.transpose(C))
            PPP.Sxx=SP[0][0]
            PPP.Syy=SP[1][1]
            PPP.Sxy=SP[0][1]
            PPP.plot(self.ax, "Prat")
            PPP.ErrorElipse()
            PointCSV.append([PPP.my_id,self.Meshiha_Table.at[i,"Code"],PPP.my_x,PPP.my_y,PPP.Sxx,PPP.Syy,PPP.Sxy,PPP.Smax,PPP.Smin,PPP.phi*180/np.pi])
        self.canvas.draw()
        np.savetxt(f"Pratim_meshiha_errors.csv", np.array(PointCSV), delimiter=",", fmt='%s')



    def Load(self):
        self.azimuth=[]
        self.Points=[]

        self.ControlPoints()
        self.Sla=pd.read_csv("Sla.csv",header=None).to_numpy()
        self.Sx=pd.read_csv("Sx.csv",header=None).to_numpy()
        self.Lb=pd.read_csv("Lb.csv",header=None).to_numpy()
        XX=pd.read_csv("Xa.csv")
        Ellipsee=pd.read_csv("Error_ellipses.csv")
        for i in range(len(XX.iloc[:,0])):
            h=0
            sh=0
            HHH=0
            try:
                h=XX.at[i, 'H']
                sh=XX.at[i, 'Height Accuracy']
            except KeyError:
                HHH=np.inf
                pass
            self.Points.append(Point2D(XX.at[i, 'X'],XX.at[i, 'Y'],XX.at[i, 'Name'],H=h,Sxx=Ellipsee.at[i, 'Sxx'],Syy=Ellipsee.at[i, 'Syy'],Sxy=Ellipsee.at[i, 'Sxy'],Sh=sh))
            self.Points[-1].ErrorElipse()
            self.Points[-1].plot(self.ax)
            self.canvas.draw()
        if HHH == np.inf:
            messagebox.showinfo('there is no height', 'The poind louded without height value')
        All_azimuths=pd.read_csv("All_azimuths.csv",header=None)
        for AZ in All_azimuths.iloc[0,:]:
            self.azimuth.append(AZ)

        with open('Tzloun_Accuracy.txt') as f:
            lines = f.readlines()
        self.Sigapost=float(lines[1][14:])


    def Quit(self):  # Close the program
        self.top.quit()
        self.top.destroy()