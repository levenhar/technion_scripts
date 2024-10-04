
import numpy as np
import MatrixMethods as MM


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


    def ImageCorner(self):
        Corner=np.array([[self.SensorSize/2,self.SensorSize/2,-self.focalLength],
                         [-self.SensorSize/2,self.SensorSize/2,-self.focalLength],
                         [-self.SensorSize/2,-self.SensorSize/2,-self.focalLength],
                         [self.SensorSize/2,-self.SensorSize/2,-self.focalLength]])
        return Corner




