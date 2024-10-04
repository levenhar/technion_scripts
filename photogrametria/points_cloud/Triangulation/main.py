import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Triangulation import Triangulation

Data = pd.read_csv("data2.txt", sep='\s+', header=None, names=['x', 'y', 'z'])
Dataarray = Data.to_numpy()

TriagolationArray = Triangulation(Dataarray)

CH = TriagolationArray.ConvexHull()

for T in TriagolationArray.Triangles:
    T.plot(False,"black",Dataarray)

plt.plot(CH[:,0], CH[:,1], c="blue")

Point = np.array([[6, 119, 0]]).T

H, TT= TriagolationArray.PointHight(Point)

TT.plot(False,"red",Dataarray)
plt.scatter(Point[0,:],Point[1,:])

plt.text(Point[0,:],Point[1,:],f"H = {round(H,3)}")
plt.show()

Point2 = np.array([[12.8,115.7,0]]).T
Section, Points = TriagolationArray.HightSection(Point,Point2)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for T in TriagolationArray.Triangles:
    T.plot(False,"black",Dataarray,ax1)


L = ((Point - Point2).T@(Point - Point2))[0,0]
ax1.plot(Points[:,0],Points[:,1], c="b")
ax1.scatter(Points[:,0],Points[:,1])
ax1.text(Points[0,0],Points[0,1],"Start Point")
ax1.text(Points[-1,0],Points[-1,1],"End Point")

ax2.plot(Section[:,0]*L,Section[:,1])
plt.show()

TriagolationArray.plot()

Vetex2Constraint = pd.read_csv("constrains.txt", header=None, names=["1","2"]).to_numpy().astype(int)

for V2C in Vetex2Constraint:
    TriagolationArray = TriagolationArray.AddConstraints(list(V2C))
    for T in TriagolationArray.Triangles:
        T.plot(False,"black",Dataarray)
    plt.plot(Dataarray[V2C, 0], Dataarray[V2C, 1])
    plt.show()


#########  create hight section with the Constraint #########################################

Point1 = np.array([[7.5,120.5,0]]).T
Point2 = np.array([[13.5,115.5,0]]).T

Section, Points = TriagolationArray.HightSection(Point1,Point2)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for T in TriagolationArray.Triangles:
    T.plot(False,"black",Dataarray,ax1)


L = ((Point1 - Point2).T@(Point1 - Point2))[0,0]
ax1.plot(Points[:,0],Points[:,1], c="b")
ax1.scatter(Points[:,0],Points[:,1])
ax1.text(Points[0,0],Points[0,1],"Start Point")
ax1.text(Points[-1,0],Points[-1,1],"End Point")

ax2.plot(Section[:,0]*L,Section[:,1])
plt.show()

x = 9
