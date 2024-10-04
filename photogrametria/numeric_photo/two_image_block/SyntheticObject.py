import numpy as np
from Camera import Camera
from Point2D import ImagePoints
from Point3D import GroundPoint
from SingleImage import SingleImage
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

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
            Image1.TiePoints.append(f"{type}{i}")
        elif type == "GCP":
            Image1.GCP.append(f"{type}{i}")
    Groundpoints = []
    Point_2 = []
    removeList = []
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
            T2.Images.append("01")
            T2.Images.append("02")
            Point_2.append(T2)

            if type == "Tie":
                Image2.TiePoints.append(G.id)
            elif type == "GCP":
                Image2.GCP.append(G.id)
            Groundpoints[i].Images.append(Image2.id)
        elif type == "Tie":
            removeList.append(i)

    Point_1 = np.delete(Point_1, removeList)
    Groundpoints = np.delete(Groundpoints, removeList)
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
    Image1 = SingleImage(C1, "01")
    orintationS = []
    for i in range(3):
        orintationS.append(1*np.pi/180 * np.random.random())
    Image1.exteriorOrientationParameters = {"X0": -B / 2, "Y0": 0, "Z0": H, "omega": orintationS[0],
                                            "phi": orintationS[1], "kappa": orintationS[2]}

    orintationS = []
    for i in range(3):
        orintationS.append(1*np.pi/180 * np.random.random())
    Image2 = SingleImage(C1, "02")
    Image2.exteriorOrientationParameters = {"X0": B / 2, "Y0": 0, "Z0": H, "omega": orintationS[0],
                                            "phi": orintationS[1], "kappa": orintationS[2]}
    ImageList = [Image1, Image2]
    OriantationList = [Image1.exteriorOrientationParameters, Image2.exteriorOrientationParameters]
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
                ImageTiePointsList[p1] = np.concatenate((ImageTiePointsList[p1],TiePoint_1))
                ImageTiePointsList[p2] = np.concatenate((ImageTiePointsList[p2],TiePoint_2))
                GroundTiepointsList = np.concatenate((GroundTiepointsList,GroundTiepoints))

    if tamplate == "firstImage" or tamplate == "Overlap Area":
        if tamplate == "firstImage":
            ImageControlPoints1 = s * np.random.random((n, 2)) - s / 2

        elif tamplate == "Overlap Area":
            cor = np.array([[s / 2, s / 2], [s / 2, -s / 2], [-s / 2, -s / 2], [-s / 2, s / 2]])
            Poly1 = Polygon(cor)
            GP = np.array(list(map(lambda C: Image2.ImageToGround_GivenZ(C, 0), cor)))
            Cor2 = np.array(list(map(lambda P: Image1.GroundToImage(P), GP))).reshape((4, 2))
            Poly2 = Polygon(Cor2)
            intersection_poly = Poly1.intersection(Poly2)
            min_x, min_y, max_x, max_y = intersection_poly.bounds
            ImageControlPoints1 = []
            while len(ImageControlPoints1) < n:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                P = Point(x,y)
                if intersection_poly.contains(P):
                    ImageControlPoints1.append([x,y])
            ImageControlPoints1 = np.array(ImageControlPoints1)

        ControlPoint_1, ControlPoint_2, GroundControlPoints = homogeticPoints(ImageControlPoints1, Image1, Image2, "GCP",0,0)
        ImageColntrolPointsList = [ControlPoint_1, ControlPoint_2]
    cor = np.array([[s / 2, s / 2], [s / 2, -s / 2], [-s / 2, -s / 2], [-s / 2, s / 2]])
    GP1 = np.array(list(map(lambda C: Image1.ImageToGround_GivenZ(C, 0), cor)))
    GP2 = np.array(list(map(lambda C: Image2.ImageToGround_GivenZ(C, 0), cor)))

    if tamplate == "Corner":
        SPoly1 = Polygon(GP1)
        SPoly2 = Polygon(GP2)
        combination = SPoly1.union(SPoly2)

        centroid = np.array(combination.centroid.coords.xy).T
        centroid = np.concatenate((centroid,np.array([[0]])), axis=1).reshape(-1,)
        GCP = np.array([centroid, [centroid[0]+2*B/p,centroid[1]+1.5*B/p, 0], [centroid[0]+2*B/p, centroid[1]-1.5*B/p,0],[centroid[0]-2*B/p,centroid[1]+1.5*B/p, 0], [centroid[0]-2*B/p, centroid[1]-1.5*B/p,0]])

        GroundControlPoints = []
        for g in range(len(GCP)):
            GroundControlPoints.append(GroundPoint(GCP[g,0], GCP[g,1], GCP[g,2], "GCP", f"GCP{g}"))

        Point_1 = []
        Point_2 = []
        for G in GroundControlPoints:
            Point1 = Image1.GroundToImage(np.array([G.X, G.Y, G.Z]))
            Point2 = Image2.GroundToImage(np.array([G.X, G.Y, G.Z]))
            if -Image1.camera.SensorSize / 2 <= Point1[0, 0] <= Image1.camera.SensorSize / 2 and -Image1.camera.SensorSize / 2 <= Point1[0, 1] <= Image1.camera.SensorSize / 2:
                T1 = ImagePoints(G.id, Point1[0, 0], Point1[0, 1], "Control", G)
                T1.Images.append("01")
                Point_1.append(T1)
            if -Image1.camera.SensorSize / 2 <= Point2[0, 0] <= Image1.camera.SensorSize / 2 and -Image1.camera.SensorSize / 2 <= Point2[0, 1] <= Image1.camera.SensorSize / 2:
                T2 = ImagePoints(G.id, Point2[0, 0], Point2[0, 1], "Control", G)
                T2.Images.append("02")
                Point_2.append(T2)

        ImageColntrolPointsList = [Point_1, Point_2]

    plt.plot(GP1[:, 0], GP1[:, 1],c="g")
    plt.plot([GP1[0, 0],GP1[-1, 0]],[GP1[0, 1],GP1[-1, 1]],c="g")

    plt.plot(GP2[:, 0], GP2[:, 1],c="m")
    plt.plot([GP2[0, 0], GP2[-1, 0]], [GP2[0, 1], GP2[-1, 1]],c="m")
    for i in range(len(GroundTiepointsList)):
        GroundTiepointsList[i].plot_XY("b",".")
    for i in range(len(GroundControlPoints)):
        GroundControlPoints[i].plot_XY("r","^")
    plt.axis("equal")
    plt.show()
    ############### save in csv #########################################################################################################################
    GTP=np.zeros((len(GroundTiepointsList),3))
    for i in range(len(GroundTiepointsList)):
        GTP[i,:] = np.array(GroundTiepointsList[i].convert2list())[1:].astype(np.float)

    GCP = np.zeros((len(GroundControlPoints), 3))
    for i in range(len(GroundControlPoints)):
        GCP[i, :] = np.array(GroundControlPoints[i].convert2list())[1:].astype(np.float)

    np.savetxt('GroundTiepoints.csv', GTP, delimiter=',')
    np.savetxt('GroundControlPoints.csv', GCP, delimiter=',')

    for img in range(len(ImageTiePointsList)):
        imgPoints=np.zeros((len(ImageTiePointsList[img]), 3)).astype(str)
        for i in range(len(ImageTiePointsList[img])):
            imgPoints[i, :] = np.array(ImageTiePointsList[img][i].convert2list())
        np.savetxt(f'ImageTiepoints_{img+1}.csv', imgPoints, delimiter=',',fmt='%s')

    for img in range(len(ImageColntrolPointsList)):
        imgPoints=np.zeros((len(ImageColntrolPointsList[img]), 3)).astype(str)
        for i in range(len(ImageColntrolPointsList[img])):
            imgPoints[i, :] = np.array(ImageColntrolPointsList[img][i].convert2list())
        np.savetxt(f'ImageControlpoints_{img+1}.csv', imgPoints, delimiter=',',fmt='%s')

    return OriantationList, GroundTiepointsList, ImageTiePointsList, ImageColntrolPointsList, GroundControlPoints
