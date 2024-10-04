import numpy as np
from Camera import Camera
import math
from numpy.linalg import norm
from Camera import Camera
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix
import cv2
import scipy


class SingleImage(object):

    def __init__(self, camera):
        """
        Initialize the SingleImage object

        :param camera: instance of the Camera class
        :param points: points in image space

        :type camera: Camera
        :type points: np.array

        """
        self.__camera = camera
        self.__innerOrientationParameters = np.array([0,1,0,0,0,1]).reshape(6,1)
        self.__isSolved = False
        self.__exteriorOrientationParameters = np.array([0, 0, 0, 0, 0, 0], 'f')
        self.__rotationMatrix = None

    @property
    def innerOrientationParameters(self):
        """
        Inner orientation parameters


        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision

        :return: inner orinetation parameters

        :rtype: array
        """


        return self.__innerOrientationParameters

    @property
    def camera(self):
        """
        The camera that took the image

        :rtype: Camera

        """
        return self.__camera

    @property
    def exteriorOrientationParameters(self):
        r"""
        Property for the exterior orientation parameters

        :return: exterior orientation parameters in the following order, **however you can decide how to hold them (dictionary or array)**

        .. math::
            exteriorOrientationParameters = \begin{bmatrix} X_0 \\ Y_0 \\ Z_0 \\ \omega \\ \varphi \\ \kappa \end{bmatrix}

        :rtype: np.ndarray or dict
        """
        return self.__exteriorOrientationParameters

    @exteriorOrientationParameters.setter
    def exteriorOrientationParameters(self, parametersArray):
        r"""

        :param parametersArray: the parameters to update the ``self.__exteriorOrientationParameters``

        **Usage example**

        .. code-block:: py

            self.exteriorOrintationParameters = parametersArray

        """
        self.__exteriorOrientationParameters = parametersArray

    @property
    def rotationMatrix(self):
        """
        The rotation matrix of the image

        Relates to the exterior orientation
        :return: rotation matrix

        :rtype: np.ndarray (3x3)
        """

        R = Compute3DRotationMatrix(self.exteriorOrientationParameters["omega"], self.exteriorOrientationParameters["phi"],
                                    self.exteriorOrientationParameters["kappa"])

        return R

    @property
    def isSolved(self):
        """
        True if the exterior orientation is solved

        :return True or False

        :rtype: boolean
        """
        return self.__isSolved

    @isSolved.setter
    def isSolved(self,boolValue):
        self.__isSolved=boolValue

    def ComputeInnerOrientation(self, imagePoints,Error):
        r"""
        Compute inner orientation parameters

        :param imagePoints: coordinates in image space

        :type imagePoints: np.array nx2

        :return: Inner orientation parameters, their accuracies, and the residuals vector

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            - Don't forget to update the ``self.__innerOrinetationParameters`` member. You decide the type
            - The fiducial marks are held within the camera attribute of the object, i.e., ``self.camera.fiducialMarks``
            - return values can be a tuple of dictionaries and arrays.

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            inner_parameters, accuracies, residuals = img.ComputeInnerOrientation(img_fmarks)
        """
        ### camera fidushels
        fiduXcam = self.__camera.fiducialMarks[:, 0]
        fiduYcam = self.__camera.fiducialMarks[:, 1]

        ### Image fidushels
        fiduXimg = imagePoints[:, 0]
        fiduYimg = imagePoints[:, 1]

        ### if Error = -1 there is not outliers, other the value of Error mark the number of the outlier observation
        if Error !=-1:
            fiduXcam=np.delete(fiduXcam,Error,0)
            fiduYcam = np.delete(fiduYcam, Error, 0)
            fiduXimg = np.delete(fiduXimg, Error, 0)
            fiduYimg = np.delete(fiduYimg, Error, 0)
        A = np.ones((2 * len(fiduXimg), 6))  # initiate A metrix
        L = np.zeros((2 * len(fiduXimg), 1))  # initiate L vector
        c = 0
        for i in range(0, 2 * len(fiduXimg), 2):
            L[i, 0] = fiduXimg[c]
            L[i + 1, 0] = fiduYimg[c]
            A[i, :] = [1, fiduXcam[c], fiduYcam[c], 0, 0, 0]
            A[i + 1, :] = [0, 0, 0, 1, fiduXcam[c], fiduYcam[c]]
            c += 1

        ### linear adusment prosses
        N = np.matmul(A.T, A)
        U = np.matmul(A.T, L)
        X = np.matmul(np.linalg.inv(N), U)
        self.__innerOrientationParameters = X
        V = np.matmul(A, X) - L
        S = np.matmul(V.T,V) / (2 * len(fiduXimg) - 6)
        sigma = np.sqrt(round(S[0, 0], 5))
        SigmaX = sigma**2 * np.linalg.inv(N)

        np.savetxt("InnerOrientation.csv", X, delimiter=",")
        np.savetxt("V.csv", V, delimiter=",")
        np.savetxt("SigmaAposteriori.csv", S, delimiter=",")
        np.savetxt("SigmaX.csv", SigmaX, delimiter=",")

        return {"InnerOrientation":X,"accuracies":SigmaX,"residuals vector":V}


    def ComputeGeometricParameters(self):
        """
        Computes the geometric inner orientation parameters

        :return: geometric inner orientation parameters

        :rtype: dict

        .. warning::

           This function is empty, need implementation

        .. note::

            The algebraic inner orinetation paramters are held in ``self.innerOrientatioParameters`` and their type
            is according to what you decided when initialized them

        """

        a0=self.__innerOrientationParameters[0][0]
        a1=self.__innerOrientationParameters[1][0]
        a2=self.__innerOrientationParameters[2][0]
        b0=self.__innerOrientationParameters[3][0]
        b1=self.__innerOrientationParameters[4][0]
        b2=self.__innerOrientationParameters[5][0]


        rotationAngle=math.atan2(b1,b2)
        translationY=b0
        translationX=a0
        shearAngle=math.atan2((a1*np.sin(rotationAngle)+a2*np.cos(rotationAngle)),(b1*np.sin(rotationAngle)+b2*np.cos(rotationAngle)))
        scaleFactorY =(a1 * np.sin(rotationAngle)+a2*np.cos(rotationAngle))/np.sin(shearAngle)
        scaleFactorX = a1 * np.cos(rotationAngle)-a2*np.sin(rotationAngle)

        return {"rotationAngle":rotationAngle*180/np.pi,"translationY":translationY,"translationX":translationX,"shearAngle":shearAngle*206265,"scaleFactorY":scaleFactorY,"scaleFactorX":scaleFactorX}

    def ComputeInverseInnerOrientation(self):
        """
        Computes the parameters of the inverse inner orientation transformation

        :return: parameters of the inverse transformation

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation algebraic parameters are held in ``self.innerOrientationParameters``
            their type is as you decided when implementing
        """

        X=self.__innerOrientationParameters
        R=np.array([[X[1][0],X[4][0]],[X[2][0],X[5][0]]])
        Rt=np.linalg.inv(R)
        T=np.array([[X[0][0]],[X[3][0]]])
        Tt=Rt@(-T)

        return {"c0":Tt[0][0],"c1":Rt[0][0],"c2":Rt[0][1],"d0":Tt[1][0],"d1":Rt[1][0],"d2":Rt[1][1]}
    def CameraToImage(self, cameraPoints):
        """
        Transforms camera points to image points

        :param cameraPoints: camera points

        :type cameraPoints: np.array nx2

        :return: corresponding Image points

        :rtype: np.array nx2


        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_image = img.Camera2Image(fMarks)

        """
        CP=cameraPoints.T
        X = self.__innerOrientationParameters
        R = np.array([[X[1][0], X[4][0]], [X[2][0], X[5][0]]])
        T = np.array([[X[0][0]], [X[3][0]]])

        IP=T+R@CP

        return IP.T


    def ImageToCamera(self, imagePoints):
        """

        Transforms image points to ideal camera points

        :param imagePoints: image points

        :type imagePoints: np.array nx2

        :return: corresponding camera points

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``


        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_camera = img.Image2Camera(img_fmarks)

        """
        IP = imagePoints.T
        X = self.ComputeInverseInnerOrientation()
        R = np.array([[X["c1"], X["c2"]], [X["d1"], X["d2"]]])
        T = np.array([[X["c0"]], [X["d0"]]])

        CP = T + R @ IP

        return CP.T

    '''invParams = self.ComputeInverseInnerOrientation()  # Getting the inverse inner orientation parameters
    camPnoints = np.array([[0.0, 0]] * len(imagePoints))  # Array of the calculated image points
    for i in range(len(imagePoints)):  # Calculation of the transformation from image points to camera points
        camPnoints[i][0] = invParams['c1'] * (invParams['c0'] + imagePoints[i][0]) + invParams['c2'] * (
                invParams['c0'] + imagePoints[i][1])
        camPnoints[i][1] = invParams['d1'] * (invParams['d0'] + imagePoints[i][0]) + invParams['d2'] * (
                invParams['d0'] + imagePoints[i][1])
    return camPnoints'''

    def ComputeExteriorOrientation(self, imagePoints, groundPoints, epsilon):
        """
        Compute exterior orientation parameters.

        This function can be used in conjecture with ``self.__ComputeDesignMatrix(groundPoints)`` and ``self__ComputeObservationVector(imagePoints)``

        :param imagePoints: image points
        :param groundPoints: corresponding ground points

            .. note::

                Angles are given in radians

        :param epsilon: threshold for convergence criteria

        :type imagePoints: np.array nx2
        :type groundPoints: np.array nx3
        :type epsilon: float

        :return: Exterior orientation parameters: (X0, Y0, Z0, omega, phi, kappa), their accuracies, and residuals vector. *The orientation parameters can be either dictionary or array -- to your decision*

        :rtype: dict

        .. warning::

           - This function is empty, need implementation
           - Decide how the parameters are held, don't forget to update documentation

        .. note::

            - Don't forget to update the ``self.exteriorOrientationParameters`` member (every iteration and at the end).
            - Don't forget to call ``cameraPoints = self.ImageToCamera(imagePoints)`` to correct the coordinates              that are sent to ``self.__ComputeApproximateVals(cameraPoints, groundPoints)``
            - return values can be a tuple of dictionaries and arrays.

        **Usage Example**

        .. code-block:: py

            img = SingleImage(camera = cam)
            grdPnts = np.array([[201058.062, 743515.351, 243.987],
                        [201113.400, 743566.374, 252.489],
                        [201112.276, 743599.838, 247.401],
                        [201166.862, 743608.707, 248.259],
                        [201196.752, 743575.451, 247.377]])
            imgPnts3 = np.array([[-98.574, 10.892],
                         [-99.563, -5.458],
                         [-93.286, -10.081],
                         [-99.904, -20.212],
                         [-109.488, -20.183]])
            img.ComputeExteriorOrientation(imgPnts3, grdPnts, 0.3)


        """
        #CameraPoints = self.ImageToCamera(imagePoints) #convert to camera coordinate system
        CameraPoints=imagePoints
        Lb=np.array([])  # initiate Lb vector
        for i in range(len(CameraPoints)):
            Lb=np.append(Lb,CameraPoints[i,0])
            Lb = np.append(Lb, CameraPoints[i,1])
        self.__ComputeApproximateVals(CameraPoints, groundPoints) #calculate  Approximate Values
        newVal=np.array([self.exteriorOrientationParameters["X0"],self.exteriorOrientationParameters["Y0"],self.exteriorOrientationParameters["Z0"],self.exteriorOrientationParameters["omega"],self.exteriorOrientationParameters["phi"],self.exteriorOrientationParameters["kappa"]])
        oldval=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
        n=0

        ### un-linear adusment prosses
        while np.linalg.norm(newVal-oldval)>epsilon:
            oldval=newVal

            A=self.__ComputeDesignMatrix(groundPoints)
            L0=self.__ComputeObservationVector(groundPoints)

            L=Lb-L0

            ### if f=0 than there is not adusment prosses
            if A.shape[0]==A.shape[1]:
                N=np.copy(A)
                U=np.copy(L)
            else:
                N=A.T@A
                U=A.T@L
            dX=np.linalg.inv(N)@U

            self.__exteriorOrientationParameters["X0"] = oldval[0] + dX[0]
            self.__exteriorOrientationParameters["Y0"] = oldval[1] + dX[1]
            self.__exteriorOrientationParameters["Z0"] = oldval[2] + dX[2]
            self.__exteriorOrientationParameters["omega"] = oldval[3] + dX[3]
            self.__exteriorOrientationParameters["phi"] =oldval[4] + dX[4]
            self.__exteriorOrientationParameters["kappa"] = oldval[5] + dX[5]
            newVal = np.array([self.exteriorOrientationParameters["X0"], self.exteriorOrientationParameters["Y0"],
                               self.exteriorOrientationParameters["Z0"], self.exteriorOrientationParameters["omega"],
                               self.exteriorOrientationParameters["phi"], self.exteriorOrientationParameters["kappa"]])
            n+=1

        ###Acurresy calculation
        V = A @ dX - L
        f=(2*len(imagePoints) - 6)
        ### if f=0 than there is not adusment prosses
        if f>0:
            sigApost = (V.T @ V) / f
            SigX = np.sqrt(np.diag((sigApost * np.linalg.inv(N))))
            SigX[3:]=SigX[3:]*206265

        else:
            SigX=None


        self.__isSolved=True

        return self.exteriorOrientationParameters, SigX, V


    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        O=np.array([self.exteriorOrientationParameters["X0"],self.exteriorOrientationParameters["Y0"],self.exteriorOrientationParameters["Z0"]])
        XYZ = (self.rotationMatrix.T @ (groundPoints-O).T).T
        Campoints=np.zeros((len(groundPoints),2))
        for n in range(len(groundPoints)):
            Campoints[n,0]=-self.camera.focalLength*(XYZ[n,0])/(XYZ[n,2])
            Campoints[n,1] = -self.camera.focalLength*(XYZ[n,1])/(XYZ[n,2])

        #imgPoints=self.CameraToImage(Campoints)
        return Campoints


    def ImageToRay(self, imagePoints):
        """
        Transforms Image point to a Ray in world system

        :param imagePoints: coordinates of an image point

        :type imagePoints: np.array nx2

        :return: Ray direction in world system

        :rtype: np.array nx3

        .. warning::

           This function is empty, need implementation

        .. note::

            The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
        """
        pass  # delete after implementations

    def ImageToGround_GivenZ(self, imagePoints, Z_values):
        """
        Compute corresponding ground point given the height in world system

        :param imagePoints: points in image space
        :param Z_values: height of the ground points


        :type Z_values: np.array nx1
        :type imagePoints: np.array nx2
        :type eop: np.ndarray 6x1

        :return: corresponding ground points

        :rtype: np.ndarray

        .. warning::

             This function is empty, need implementation

        .. note::

            - The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
            - The focal length can be called by ``self.camera.focalLength``

        **Usage Example**

        .. code-block:: py


            imgPnt = np.array([-50., -33.])
            img.ImageToGround_GivenZ(imgPnt, 115.)

        """
        CamP=self.ImageToCamera(imagePoints) #convert to camera coordinate system
        R=self.rotationMatrix
        groundP=np.array([])
        for n in range(len(CamP)):
            V=R@np.array([CamP[n,0],CamP[n,1],-self.camera.focalLength])  #calculate the ray diractin in the world system
            lambdaa=(Z_values-self.exteriorOrientationParameters["Z0"])/V[2]
            X_value=self.exteriorOrientationParameters["X0"]+lambdaa*V[0]
            Y_value = self.exteriorOrientationParameters["Y0"] + lambdaa * V[1]
            groundP=np.append(groundP,[round(X_value,3),round(Y_value,3)])


        return groundP

    def findVanishingPoint(self, points_line1, point_line2):
        P1 = points_line1 #np.concatenate((points_line1, np.ones((len(points_line1),1))),axis=1)
        P2 = point_line2 #np.concatenate((point_line2, np.ones((len(point_line2),1))),axis=1)
        nl1 = (self.__computeLineNormal(P1[::2,:],P1[1::2,:])).reshape((1,3))
        nl2 = (self.__computeLineNormal(P2[::2, :], P2[1::2, :])).reshape((1,3))

        nl1 = nl1 / nl1[0,2]
        nl2 = nl2 / nl2[0, 2]

        V = self.__computeLineNormal(nl1, nl2)
        V = V / V[2]
        return V.reshape((3,1))

    def ProjectiveCanclation(self,line1_1, Line1_2, Line2_1, Line2_2, ImageMatrix):
        line1_1 = np.concatenate((line1_1,np.ones((len(line1_1),1))),axis=1)
        Line1_2 = np.concatenate((Line1_2,np.ones((len(Line1_2),1))),axis=1)
        Line2_1 = np.concatenate((Line2_1,np.ones((len(Line2_1),1))),axis=1)
        Line2_2 = np.concatenate((Line2_2,np.ones((len(Line2_2),1))),axis=1)
        V1 = self.findVanishingPoint(line1_1, Line1_2)
        V2 = self.findVanishingPoint(Line2_1, Line2_2)
        HorizonLine = self.__computeLineNormal(V1.T, V2.T)
        HorizonLine = HorizonLine / HorizonLine[2]
        H = np.eye(3)
        H[2, :] = HorizonLine.reshape((1, 3))

        NewImage = cv2.warpPerspective(ImageMatrix, H, (ImageMatrix.shape[0]//5,ImageMatrix.shape[1]//5))
        NewImage = cv2.cvtColor(NewImage, cv2.COLOR_BGR2RGB)

        return NewImage

    # --------------------- 3D reconstruction -------------------------------------
    def __computeLineNormal(self, points1, points2):
        """
        Compute the normal of interpretation plane defined by two points and the perspective center

        :param points1: homogeneous coordinates of the first points set
        :param points2: homogeneous coordinates of the second points set

        :type points1: np.array nx3
        :type points2: np.array nx3

        :return: normal of the interpretation plane defined by the two points

        :rtype: np.array nx3

        """

        # check if the points are normalized
        if np.all(norm(points1, axis=1) != 1) or np.all(norm(points2, axis=1) != 1):
            points1 = points1 / np.linalg.norm(points1, axis=1).reshape((len(points1),1))
            points2 = points2 / np.linalg.norm(points2, axis=1).reshape((len(points2),1))

        # Compute the normal of the interpretation plane
        nl = np.average(np.cross(points1, points2),axis=0)

        return nl

    def faceNormal_imageSpace(self, r1, r2):
        """
        Compute face normal in image space

        :param r1: vector in the first direction
        :param r2: vector in the second direction

        :type r1: np.array 1x3
        :type r2: np.array 1x3

        :return: face normal
        :rtype: np.array 1x3

        .. warning ::

            Empty. needs implementation.

        """
        pass

    def faceNormal_objectSpace(self, n):
        """
        Compute the normal in object space

        :param n: normal in image space

        :type n: np.array 1x3

        :return: normal in object space
        :rtype: np.array 1x3

        .. warning ::

            Empty. needs implementation.

        """
        pass

    def scale_firstFace(self, normal, s, x1, x2):
        """
        Compute the distance of the first normal

        :param normal: face normal in object space
        :param s: known measure in object space
        :param x1: point in image space
        :param x2: point in image space

        :type s: float
        :type x1: np.array 1x3
        :type x2: np.array 1x3

        :return: distance of the first plane from the perspective center in object space

        :rtype: float

        .. warning ::

            Empty. needs implementation.
        """
        pass

    def rotationMatrix_vanishingPoints(self, vanishingPoint1, vanishingPoint2, vanishingPoint3):
        """
        Compute rotation matrix according to three computed vanishing points that define three cardinal directions

        :param vanishingPoint1: vanishing point
        :param vanishingPoint2: vanishing point

        :type vanishingPoint1: np.array 1x3
        :type vanishingPoint2: np.array 1x3

        :return: rotation matrix
        :rtype: np.array 3x3

        """
        x1, y1 = vanishingPoint1[0], vanishingPoint1[1]
        x2, y2 = vanishingPoint2[0], vanishingPoint2[1]
        x3, y3 = vanishingPoint3[0], vanishingPoint3[1]

        A = np.array([[-(x3-x2),-(y3,y2)],[-(x3-x1), -(y3-y1)]])
        b = np.array([[-(x3-x2)*x1-(y3-y2)*y1, -(x3-x1)*x2-(y3-y1)*y2]]).T

        X = np.linalg.inv(A)@b
        f = np.sqrt(-(x1-X[0])*(x2-X[0])-(y1-X[1])*(y2-X[1]))

        L1 = f/ (np.sqrt(x1-X[0])**2+(y1-X[1])**2+f**2)
        L2 = f/ (np.sqrt(x2-X[0])**2+(y2-X[1])**2+f**2)
        L3 = f/ (np.sqrt(x3-X[0])**2+(y3-X[1])**2+f**2)


        LL = np.eye(3)
        LL[0, 0] = L1
        LL[1, 1] = L2
        LL[2, 2] = L3

        R = LL@np.array([[-1/f*(x1-X[0]), -1/f*(y1-X[1]), 1],[-1/f*(x2-X[0]), -1/f*(y2-X[1]), 1], [-1/f*(x3-X[0]), -1/f*(y3-X[1]), 1]])

        self.rotationMatrix = R
        return R


    def mapPoints(self, normal, distances, points):
        """

        :param normal: face normal object space
        :param distances:  face distance in object space
        :param points: homogeneous coordinates of points in image space

        :type normal: np.array 1x3
        :type distances: float
        :type points: np.array nx3

        :return: points in object space, scale for the next face

        :rtype: tuple

        """

        # Map points to object space
        points_normalized = points.T / norm(points, axis=1)  # normalize points in image space
        distances = distances / np.dot(points, normal.T)  # 1D array
        points_objectSpace = (distances * points_normalized.T)  # multiplication of 1D array and a matrix

        # Compute a known measure in the second face
        s = norm(points_objectSpace[0] - points_objectSpace[1])

        return points_objectSpace, s

    # ---------------------- Private methods ----------------------

    def __ComputeApproximateVals(self, cameraPoints, groundPoints):
        """
        Compute exterior orientation approximate values via 2-D conform transformation

        :param cameraPoints: points in image space (x y)
        :param groundPoints: corresponding points in world system (X, Y, Z)

        :type cameraPoints: np.ndarray [nx2]
        :type groundPoints: np.ndarray [nx3]

        :return: Approximate values of exterior orientation parameters
        :rtype: np.ndarray or dict

        .. note::

            - ImagePoints should be transformed to ideal camera using ``self.ImageToCamera(imagePoints)``. See code below
            - The focal length is stored in ``self.camera.focalLength``
            - Don't forget to update ``self.exteriorOrientationParameters`` in the order defined within the property
            - return values can be a tuple of dictionaries and arrays.

        .. warning::

           - This function is empty, need implementation
           - Decide how the exterior parameters are held, don't forget to update documentation

        """

        # Find approximate values
        phi=0
        omega=0

        L=np.array([])
        A=np.zeros((4,4))
        for i in [0,2]:
            L=np.append(L,groundPoints[i,0])
            L = np.append(L, groundPoints[i, 1])
            A[i,:]=np.array([1,0,cameraPoints[i,0],cameraPoints[i,1]])
            A[i+1, :] = np.array([0, 1, -cameraPoints[i, 1], cameraPoints[i, 0]])

        L=L.T
        X=np.linalg.inv(A)@L

        X0=X[0]
        Y0=X[1]
        k=np.arctan2(-X[3], X[2])
        lambd=np.sqrt(X[2]**2+X[3]**2)

        Z0=groundPoints[i, 2]+lambd*self.camera.focalLength

        self.exteriorOrientationParameters={"X0":X0,"Y0":Y0,"Z0":Z0,"omega":omega,"phi":phi,"kappa":k}

    def __ComputeObservationVector(self, groundPoints):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: Vector l0

        :rtype: np.array nx1
        """

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:,0] - self.exteriorOrientationParameters["X0"]
        dY = groundPoints[:,1] - self.exteriorOrientationParameters["Y0"]
        dZ = groundPoints[:,2] - self.exteriorOrientationParameters["Z0"]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix.T, dXYZ).T

        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = -self.camera.focalLength * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] = -self.camera.focalLength * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

        return l0

    def __ComputeDesignMatrix(self, groundPoints):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

        """
        # initialization for readability
        omega = self.exteriorOrientationParameters["omega"]
        phi = self.exteriorOrientationParameters["phi"]
        kappa = self.exteriorOrientationParameters["kappa"]

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters["X0"]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters["Y0"]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters["Z0"]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = self.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = self.camera.focalLength / rT3g ** 2

        dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
        dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

        dgdX0 = np.array([-1, 0, 0], 'f')
        dgdY0 = np.array([0, -1, 0], 'f')
        dgdZ0 = np.array([0, 0, -1], 'f')

        # Derivatives with respect to X0
        dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
        dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

        # Derivatives with respect to Y0
        dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
        dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

        # Derivatives with respect to Z0
        dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
        dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

        dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
        dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
        dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

        gRT3g = dXYZ * rT3g

        # Derivatives with respect to Omega
        dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Phi
        dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
                                        rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
                                        rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Kappa
        dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        # all derivatives of x and y
        dd = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], 6))
        a[0::2] = dd[0]
        a[1::2] = dd[1]

        return a

    def IamgeSignature(self):
        R=self.rotationMatrix
        O1=np.array([[self.exteriorOrientationParameters["X0"]],
             [self.exteriorOrientationParameters["Y0"]],
              [self.exteriorOrientationParameters["Z0"]]])

        ImageCornner=self.camera.ImageCorner()
        Vi=R@ImageCornner.T

        #finding Scale factor for Z=0
        GroundPoints=np.array([0,0,0]).reshape(1,3)
        for i in range(4):
            L=((1-O1[2])/Vi[2,i])[0]
            XYZ=O1+L*(Vi[:,i].reshape(3,1))
            GroundPoints=np.concatenate((GroundPoints,XYZ.T),axis=0)
        GroundPoints=np.delete(GroundPoints,0,0)
        return GroundPoints

    def CornnerToCamera(self,Cornner):
        RR = self.rotationMatrix
        O = np.array([self.exteriorOrientationParameters["X0"], self.exteriorOrientationParameters["Y0"],self.exteriorOrientationParameters["Z0"]])
        XYZ = (RR.T @ (Cornner - O).T).T
        Campoints = np.zeros((len(Cornner), 2))
        for n in range(len(Cornner)):
            Campoints[n, 0] = -self.camera.focalLength * (XYZ[n, 0]) / (XYZ[n, 2])
            Campoints[n, 1] = -self.camera.focalLength * (XYZ[n, 1]) / (XYZ[n, 2])

        return Campoints









if __name__ == '__main__':
    fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
    img_fmarks = np.array([[-7208.01, 7379.35],
                           [7290.91, -7289.28],
                           [-7291.19, -7208.22],
                           [7375.09, 7293.59]])
    cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
    img = SingleImage(camera = cam)
    print(img.ComputeInnerOrientation(img_fmarks))

    print(img.ImageToCamera(img_fmarks))

    print(img.CameraToImage(fMarks))

    GrdPnts = np.array([[5100.00, 9800.00, 100.00]])
    print(img.GroundToImage(GrdPnts))

    imgPnt = np.array([23.00, 25.00])
    print(img.ImageToRay(imgPnt))

    imgPnt2 = np.array([-50., -33.])
    print(img.ImageToGround_GivenZ(imgPnt2, 115.))

    # grdPnts = np.array([[201058.062, 743515.351, 243.987],
    #                     [201113.400, 743566.374, 252.489],
    #                     [201112.276, 743599.838, 247.401],
    #                     [201166.862, 743608.707, 248.259],
    #                     [201196.752, 743575.451, 247.377]])
    #
    # imgPnts3 = np.array([[-98.574, 10.892],
    #                      [-99.563, -5.458],
    #                      [-93.286, -10.081],
    #                      [-99.904, -20.212],
    #                      [-109.488, -20.183]])
    #
    # intVal = np.array([200786.686, 743884.889, 954.787, 0, 0, 133 * np.pi / 180])
    #
    # print img.ComputeExteriorOrientation(imgPnts3, grdPnts, intVal)
