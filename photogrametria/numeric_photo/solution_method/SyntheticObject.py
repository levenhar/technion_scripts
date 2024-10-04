import numpy as np
from Camera import Camera
from Point2D import ImagePoints
from Point3D import GroundPoint
from SingleImage import SingleImage
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import MatrixMethods as MM
import PhotoViewer as PV

def homogeticPoints(TiePoint1, Image1, Image2, type, n,Inumber):
    Point_1 = []
    for i in range(len(TiePoint1)):
        if type == "Tie":
            idd=n*Inumber + i
        else:
            idd= i
        Point_1.append(ImagePoints(f"{type}{idd}", TiePoint1[i, 0], TiePoint1[i, 1], type))
        Point_1[i].Images.append(Image1.id)
        if type == "Tie":
            Image1.TiePoints.append(f"{type}{idd}")
        elif type == "GCP":
            Image1.GCP.append(f"{type}{idd}")
    Groundpoints = []
    Point_2 = []
    #removeList = []
    for i in range(len(Point_1)):
        GP = Image1.ImageToGround_GivenZ(np.array([Point_1[i].x, Point_1[i].y]), 0)
        G = GroundPoint(GP[0], GP[1], GP[2], type, Point_1[i].id)
        Point_1[i].groundPoint = G
        G.Images.append(Image1.id)
        Groundpoints.append(G)

        Point2 = Image2.GroundToImage(np.array([G.X, G.Y, G.Z]))

        if -Image1.camera.SensorSize / 2 <= Point2[0, 0] <= Image1.camera.SensorSize / 2 and -Image1.camera.SensorSize / 2 <= Point2[0, 1] <= Image1.camera.SensorSize / 2:
            Point_1[i].Images.append(Image2.id)
            T2 = ImagePoints(G.id, Point2[0, 0], Point2[0, 1], "Tie", G)
            T2.Images.append(Image1.id)
            T2.Images.append(Image2.id)
            Point_2.append(T2)

            if type == "Tie":
                Image2.TiePoints.append(G.id)
            elif type == "GCP":
                Image2.GCP.append(G.id)
            Groundpoints[i].Images.append(Image2.id)
        else:
            Point_2.append(None)

        '''elif type == "Tie":
            removeList.append(i)'''

    ''' Point_1 = np.delete(Point_1, removeList)
    Groundpoints = np.delete(Groundpoints, removeList)'''
    Point_2 = np.array(Point_2)
    return Point_1, Point_2, Groundpoints


def syntheticImages(f, s, H, p,TieNumber, tamplate, n):
    '''
    the function create the syntetic data of image Block
    :param f: the focal length
    :type f: float
    :param s: sensor size
    :type s: flost
    :param H: hight
    :type H: float
    :param p: overlap present
    :type p: float
    :param TieNumber: number of tie point in one image (can be 3 or 4)
    :type TieNumber: int
    :param tamplate: the way to order the control points
    :type tamplate: str
    :param n: number of cntrol points
    :type n: int
    :return: OriantationList, GroundTiepointsList, ImageTiePointsList, ImageColntrolPointsList, GroundControlPoints
    :rtype:list
    the function return:
    1. OriantationList - list of dictionary that contain the extiroir Oriantation
    2. GroundTiepointsList - list of all the Tie points
    3. ImageTiePointsList - lists of the Image coordinate for the tie points
    4. ImageColntrolPointsList - lists of the Image coordinate for the control points
    5. GroundControlPoints - list of the GCP coordinates
    '''
    if n < 3:
        raise Exception("you must choose more than 3 GCP")




    #### Cameras Orintations ####################################################################################################################
    C1 = Camera(f, [0, 0], [0, 0, 0], [0, 0], [], s)
    S = s * H / f  # s, f in same units (mm / pix)
    B = S * (1 - p)
    ImageList = []
    OriantationList = []
    for i in range(5):
        for j in range(5):
            Image = SingleImage(C1, f"{i+j}")

            orintationS = []
            for _ in range(3):
                orintationS.append(1*np.pi/180 * np.random.random())
            Image.exteriorOrientationParameters = {"X0": i*B, "Y0": j*B, "Z0": H, "omega": orintationS[0],
                                                    "phi": orintationS[1], "kappa": orintationS[2]}
            ImageList.append(Image)
            OriantationList.append(Image.exteriorOrientationParameters)
    ##################################################################################################################################################

    d = s / 2 - 20
    if TieNumber == 3:
        TiePoint = np.array([[0, 0], [0, d], [0, -d]])
    elif TieNumber == 4:
        TiePoint = np.array([[d, d], [-d, d], [-d, -d], [d, -d]])
    else:
        raise Exception("Number of tie Points can be 3 or 4")

    ImageTiePointsList = [np.array([]) for _ in range(len(ImageList))]

    GroundTiepointsList=np.array([])
    for p1 in range(len(ImageList)):
        for p2 in range(len(ImageList)):
            if p1 == p2:
                continue
            else:
                TiePoint_1, TiePoint_2, GroundTiepoints = homogeticPoints(TiePoint, ImageList[p1], ImageList[p2], "Tie", p1, TieNumber)
                ImageTiePointsList[p2] = np.concatenate((ImageTiePointsList[p2],TiePoint_2))

        GroundTiepointsList = np.concatenate((GroundTiepointsList,GroundTiepoints))
        ImageTiePointsList[p1] = np.concatenate((ImageTiePointsList[p1],TiePoint_1))
    Checkarray = np.array(ImageTiePointsList)
    removelist = []
    for i in range(len(Checkarray[0])):
        c=0
        for j in range((len(Checkarray))):
            if Checkarray[j,i] != None:
                c+=1
        if c<=1:
            removelist.append(i)
    GroundTiepointsList = np.delete(GroundTiepointsList, removelist)
    GroundTiepointsArray = GroundPoint.listofPoints2Array(GroundTiepointsList)
    for w in range(len(ImageTiePointsList)):
        ImageTiePointsList[w] = np.delete(ImageTiePointsList[w],removelist)


    ImageColntrolPointsList = [[] for _ in range(len(ImageList))]

    if tamplate == "firstImage":
        ImageControlPoints1 = s * np.random.random((n, 2)) - s / 2
        for k in range(1,len(ImageList)):
            ControlPoint_1, ControlPoint_2, GroundControlPoints = homogeticPoints(ImageControlPoints1, ImageList[0], ImageList[k], "GCP", 0, 0)
            ImageColntrolPointsList[k] = np.concatenate((ImageColntrolPointsList[k], ControlPoint_2))
        ImageColntrolPointsList[0] =  np.concatenate((ImageColntrolPointsList[0], ControlPoint_1))




    elif tamplate == "RandomInBlock" or tamplate == "Corner":
        polygons = np.array(list(map(lambda Img: Polygon(Img.IamgeSignature()[:, :2]), ImageList)))
        combination = unary_union(polygons)
        min_x, min_y, max_x, max_y = combination.bounds
        if tamplate == "RandomInBlock":
            GroundControlPoints = []
            while len(GroundControlPoints) < n:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                P = Point(x, y)
                if combination.contains(P):
                    GroundControlPoints.append([x, y, 0])
            GCP = np.array(GroundControlPoints)

        elif tamplate == "Corner":
            centroid = np.array(combination.centroid.coords.xy).T
            centroid = np.concatenate((centroid,np.array([[0]])), axis=1).reshape((1,3))
            ratio = 3
            Corners = np.array([[min_x+B/ratio, min_y+B/ratio, 0], [min_x + B/ratio, max_y-B/ratio, 0], [max_x - B/ratio, max_y -B/ratio, 0], [max_x - B/ratio, min_y +B/ratio, 0]])

            GCP = np.concatenate((centroid, Corners))

        GroundControlPoints = []
        for g in range(len(GCP)):
            GroundControlPoints.append(GroundPoint(GCP[g,0], GCP[g,1], GCP[g,2], "GCP", f"GCP{g}"))

        for k in range(len(ImageList)):
            for G in GroundControlPoints:
                Pointt = ImageList[k].GroundToImage(np.array([G.X, G.Y, G.Z]))
                if -ImageList[k].camera.SensorSize / 2 <= Pointt[0, 0] <= ImageList[k].camera.SensorSize / 2 and -ImageList[k].camera.SensorSize / 2 <= Pointt[0, 1] <= ImageList[k].camera.SensorSize / 2:
                    ImageColntrolPointsList[k].append(ImagePoints(f"{G.id}", Pointt[0, 0], Pointt[0, 1], "GCP"))
                        #= np.concatenate((ImageColntrolPointsList[k], Pointt.reshape(-1,)))
                else:
                    ImageColntrolPointsList[k].append(None)
                    #ImageColntrolPointsList[k] = np.concatenate((ImageColntrolPointsList[k], np.array([None, None])))

        '''for k in range(len(ImageColntrolPointsList)):
            ImageColntrolPointsList[k] = ImageColntrolPointsList[k].reshape((n,2))'''

    GroundControlpointsArray = GroundPoint.listofPoints2Array(GroundControlPoints)
    ### plot the data ##################################################################################################################################
    for Img in ImageList:
        #plt.scatter([Img.exteriorOrientationParameters["X0"]],[Img.exteriorOrientationParameters["Y0"]],[Img.exteriorOrientationParameters["Z0"]],c="Blue")
        Cornner = Img.IamgeSignature()
        Cornner = np.concatenate((Cornner,Cornner[0,:].reshape(1,3)))
        plt.plot(Cornner[:,0],Cornner[:,1])
    plt.scatter(GroundTiepointsArray[:,1].astype(np.float),GroundTiepointsArray[:,2].astype(np.float),c="blue",marker = ".")
    plt.scatter(GroundControlpointsArray[:,1].astype(np.float),GroundControlpointsArray[:,2].astype(np.float),c="red",marker = "^")
    plt.axis("equal")
    plt.show()

    return OriantationList, GroundTiepointsList, ImageTiePointsList, ImageColntrolPointsList, GroundControlPoints
