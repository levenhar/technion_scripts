import numpy as np
import MatrixMethods as MM
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, PrintMatrix
import math


class Camera(object):

    def __init__(self, focal_length, principal_point, radial_distortions, decentering_distortions, fiducial_marks,SensorSize):
        """
        Initialize the Camera object

        :param focal_length: focal length of the camera(mm)
        :param principal_point: principle point
        :param radial_distortions: the radial distortion parameters K0, K1, K2
        :param decentering_distortions: decentering distortion parameters P0, P1, P2
        :param fiducial_marks: fiducial marks in camera space

        :type focal_length: double
        :type principal_point: np.array
        :type radial_distortions: np.array
        :type decentering_distortions: np.array
        :type fiducial_marks: np.array

        """
        # private parameters
        self.__focal_length = focal_length
        self.__principal_point = principal_point
        self.__radial_distortions = radial_distortions
        self.__decentering_distortions = decentering_distortions
        self.__fiducial_marks = fiducial_marks
        self.__SensorSize = SensorSize
        self.__CalibrationParam = None
        self.__K = None

    @property
    def Calibration_Matrix(self):
        self.__K = np.array([[-self.focalLength, 0, self.principalPoint[0]],
                  [0, -self.focalLength, self.principalPoint[1]], [0, 0, 1]])

        return self.__K

    @property
    def radial_distortions(self):
        return self.__radial_distortions

    @radial_distortions.setter
    def radial_distortions(self, val):
        self.__radial_distortions = val

    @property
    def focalLength(self):
        """
        Focal length of the camera

        :return: focal length

        :rtype: float

        """
        return self.__focal_length

    @focalLength.setter
    def focalLength(self, val):
        """
        Set the focal length value

        :param val: value for setting

        :type: float

        """

        self.__focal_length = val

    @property
    def fiducialMarks(self):
        """
        Fiducial marks of the camera, by order

        :return: fiducial marks of the camera

        :rtype: np.array nx2

        """

        return self.__fiducial_marks

    @property
    def principalPoint(self):
        """
        Principal point of the camera

        :return: principal point coordinates

        :rtype: np.ndarray

        """

        return self.__principal_point

    @property
    def SensorSize(self):
        """
        Focal length of the camera

        :return: focal length

        :rtype: float

        """
        return self.__SensorSize

    @SensorSize.setter
    def SensorSize(self, val):
        """
        Set the focal length value

        :param val: value for setting

        :type: float

        """
        self.__SensorSize = val


    def Cam2Img(self, CamPoints):
        T = np.eye(3)
        T[0,2] = self.Calibration_Matrix[0,2]
        T[1, 2] = self.Calibration_Matrix[1, 2]
        CamPoints = np.concatenate((CamPoints, np.ones((len(CamPoints),1))), axis=1)
        imgPoints = T @ CamPoints.T
        imgPoints[0,:] = imgPoints[0,:] / imgPoints[2,:]
        imgPoints[1, :] = imgPoints[1, :] / imgPoints[2, :]
        return imgPoints[0:2,:].T

    def Calibration(self,CamerasPoints,ControlPoint,initioalaValue,Image):
        q = 0
        Oldvalue=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
        while np.linalg.norm(Oldvalue-initioalaValue)>0.01:
            Oldvalue=initioalaValue
            A1=self.ComputeDesignMatrix(ControlPoint,Image,initioalaValue)
            A2=np.zeros((len(ControlPoint)*2,2))
            for i in range(0,len(ControlPoint)*2,2):
                A2[i,:]=np.array([1, 0])
                A2[i+1,:]=np.array([0, 1])
            A3=np.zeros((len(ControlPoint)*2,2))
            lb = np.zeros((len(CamerasPoints)*2,1))
            l0 = self.ComputeObservationVector(ControlPoint, initioalaValue, Image)
            c=0
            for i in range(len(CamerasPoints)):
                r=np.sqrt((CamerasPoints[i,0]-initioalaValue[7])**2+(CamerasPoints[i,1]--initioalaValue[8])**2)
                xt=(CamerasPoints[i,0]-initioalaValue[7])/initioalaValue[0]
                yt=(CamerasPoints[i,1]-initioalaValue[8])/initioalaValue[0]
                A3[c, 0] = -xt * 10 ** -5 * r ** 2
                A3[c, 1] = -xt * 10 ** -10 * r ** 4
                A3[c+1, 0] = -yt * 10 ** -5 * r ** 2
                A3[c+1, 1] = -yt * 10 ** -10 * r ** 4
                l0[c] = l0[c] - xt * (10 ** -5 * initioalaValue[9] * r ** 2 + 10 ** -10 * initioalaValue[10] * r ** 4)
                l0[c+1] = l0[c+1] - yt * (10 ** -5 * initioalaValue[9] * r ** 2 + 10 ** -10 * initioalaValue[10] * r ** 4)
                lb[c, 0] = CamerasPoints[i,0]
                lb[c+1, 0] = CamerasPoints[i, 1]
                c+=2

            A=np.concatenate((A1,A2,A3),axis=1)
            L=lb.reshape(-1)-l0
            N=A.T@A
            U=A.T@L
            dx=np.linalg.inv(N)@U
            q += 1
            initioalaValue=initioalaValue+dx
        return initioalaValue



    def ComputeObservationVector(self, groundPoints,initialValue,Image):
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
        dX = groundPoints[:, 0] - Image.exteriorOrientationParameters["X0"]
        dY = groundPoints[:, 1] - Image.exteriorOrientationParameters["Y0"]
        dZ = groundPoints[:, 2] - Image.exteriorOrientationParameters["Z0"]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(Image.rotationMatrix.T, dXYZ).T

        l0 = np.empty(n * 2)

        '''xp = np.ones((len(rotated_XYZ[:, 0]))) * 
        yp = np.ones((len(rotated_XYZ[:, 0]))) * '''
        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] =initialValue[7]-initialValue[0] * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] =initialValue[8]-initialValue[0] * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

        return l0

    def compute_CalibrationMatrix(self, v1, v2, v3):
        """
        Compute the calibration parameters based on the orthocenter of the triangle
        defined by three vanishing points

        :param v1: first vanishing point
        :param v2: second vanishing point
        :param v3: third vanishing point

        :type v1: np.array
        :type v2: np.array
        :type v3: np.array

        :return: calibration matrix

        :rtype: np.array 3x3
        """

        # Solve linear system with xp and yp as unknowns

        #matrix A
        A = np.array([[v3[0] - v2[0], v3[1] - v2[1]], [v1[0] - v2[0], v1[1] - v2[1]]])
        b = np.diag(A.dot(np.array([v1, v3]).T))
        x = np.linalg.solve(A, b)

        xp = x[0]
        yp = x[1]

        # Compute the focal length
        focal = np.sqrt(- (v1 - x.flatten()).dot(v2 - x.flatten()))

        self.focalLength = focal
        self.__principal_point = np.array([xp, yp])

        return self.__K


    def CameraToIdealCamera(self, camera_points):
        """
        Transform camera coordinates to an ideal system.

        :param camera_points: set of points in camera space

        :type camera_points: np.array nx2

        :return: fixed point set

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation
        """
        pass  # delete for implementation

    def IdealCameraToCamera(self, camera_points):
        r"""
        Transform from ideal camera to camera with distortions

        :param camera_points: points in ideal camera space

        :type camera_points: np.array nx2

        :return: corresponding points in image space

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation
        """
        pass  # delete for implementation

    def ComputeDecenteringDistortions(self, camera_points):
        """
        Compute decentering distortions for given points

        :param camera_points: points in camera space

        :type camera_points: np.array nx2

        :return: decentering distortions: d_x, d_y

        :rtype: tuple of np.array

        .. warning::

            This function is empty, need implementation
        """
        pass  # delete for implementation

    def ComputeRadialDistortions(self, camera_points):
        """
        Compute radial distortions for given points

        :param camera_points: points in camera space

        :type camera_points: np.array nx2

        :return: radial distortions: delta_x, delta_y

        :rtype: tuple of np.array

        """
        pass # delete for implementation

    def CorrectionToPrincipalPoint(self, camera_points):
        """
        Correction to principal point

        :param camera_points: sampled image points

        :type: np.array nx2

        :return: corrected image points

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        .. note::

            The principal point is an attribute of the camera object, i.e., ``self.principalPoint``


        """

        pass  # Delete for implementation

    def ImageCorner(self):
        Corner=np.array([[self.SensorSize/2,self.SensorSize/2,-self.focalLength],
                         [-self.SensorSize/2,self.SensorSize/2,-self.focalLength],
                         [-self.SensorSize/2,-self.SensorSize/2,-self.focalLength],
                         [self.SensorSize/2,-self.SensorSize/2,-self.focalLength]])
        return Corner


    def Addradialerror(self,camera_points):
        k1 = self.radial_distortions[0]
        k2 = self.radial_distortions[1]

        r=(camera_points[0]**2+camera_points[1]**2)**0.5

        xt=camera_points[0]-camera_points[0]*(k1*r**2+k2*r**4)
        yt=camera_points[1]-camera_points[1]*(k1*r**2+k2*r**4)

        return [xt, yt]

    def ComputeDesignMatrix(self, groundPoints,Image,initialValue):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

        """
        # initialization for readability
        omega = initialValue[4]
        phi = initialValue[5]
        kappa = initialValue[6]

        # Coordinates subtraction
        dX = groundPoints[:, 0] - initialValue[1]
        dY = groundPoints[:, 1] - initialValue[2]
        dZ = groundPoints[:, 2] - initialValue[3]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = Image.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        dxdf = rT1g/rT3g
        dydf = rT1g/rT3g

        focalBySqauredRT3g = self.focalLength / rT3g ** 2

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
        dd = np.array([np.vstack([dxdf, dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydf, dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], 7))
        a[0::2] = dd[0]
        a[1::2] = dd[1]



        return a

if __name__ == '__main__':

    f0 = 4360.
    xp0 = 2144.5
    yp0 = 1424.5
    K1 = 0
    K2 = 0
    P1 = 0
    P2 = 0

    # define the initial values vector
    cam = Camera(f0, np.array([xp0, yp0]), np.array([K1, K2]),np.array([P1, P2]), None)
