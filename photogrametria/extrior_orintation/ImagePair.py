from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
import numpy as np


class ImagePair(object):

    def __init__(self, image1, image2):
        """
        Initialize the ImagePair class
        :param image1: First image
        :param image2: Second image
        """
        self.__image1 = image1
        self.__image2 = image2
        self.__relativeOrientationImage1 = np.array([0, 0, 0, 0, 0, 0]) # The relative orientation of the first image
        self.__relativeOrientationImage2 = None # The relative orientation of the second image
        self.__absoluteOrientation = None
        self.__isSolved = False # Flag for the relative orientation


    @property
    def isSolved(self):
        """
        Flag for the relative orientation
        returns True if the relative orientation is solved, otherwise it returns False

        :return: boolean, True or False values
        """
        return self.__isSolved

    @property
    def RotationMatrix_Image1(self):
        """
        return the rotation matrix of the first image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage1[0], self.__relativeOrientationImage1[1],
                                       self.__relativeOrientationImage1[2])

    @property
    def RotationMatrix_Image2(self):
        """
        return the rotation matrix of the second image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage2[0], self.__relativeOrientationImage2[1],
                                       self.__relativeOrientationImage2[2])

    @property
    def PerspectiveCenter_Image1(self):
        """
        return the perspective center of the first image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage1[0:3]

    @property
    def PerspectiveCenter_Image2(self):
        """
        return the perspective center of the second image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage2[0:3]

    def ImagesToGround(self, imagePoints1, imagePoints2, Method):
        """
        Computes ground coordinates of homological points

        :param imagePoints1: points in image 1
        :param imagePoints2: corresponding points in image 2
        :param Method: method to use for the ray intersection, three options exist: geometric, vector, Collinearity

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: ground points, their accuracies.

        :rtype: dict

        .. warning::

            This function is empty, need implementation


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                    [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])

            new = ImagePair(image1, image2)

            new.ImagesToGround(imagePoints1, imagePoints2, 'geometric'))

        """
        grdPointss = {}
        #store the prespective center of each image
        O1 = np.array([self.__image1.exteriorOrientationParameters["X0"],
                       self.__image1.exteriorOrientationParameters["Y0"],
                       self.__image1.exteriorOrientationParameters["Z0"]])
        O2 = np.array([self.__image2.exteriorOrientationParameters["X0"],
                       self.__image2.exteriorOrientationParameters["Y0"],
                       self.__image2.exteriorOrientationParameters["Z0"]])
        #store the rotation Matrix of ech camera
        R1=self.__image1.rotationMatrix
        R2=self.__image2.rotationMatrix
        for n in range(len(imagePoints1)):
            V1 = R1 @ np.array([imagePoints1[n, 0], imagePoints1[n, 1], -self.__image1.camera.focalLength]).T
            V2 = R2 @ np.array([imagePoints2[n, 0], imagePoints2[n, 1], -self.__image2.camera.focalLength]).T
            V1=np.reshape(V1,(3,1))
            V2 = np.reshape(V2, (3, 1))
            if Method=="geometric":
                #normalized the vectors
                V1=V1/ np.linalg.norm(V1)
                V2 = V2 / np.linalg.norm(V2)

                # calculate L vector
                L1=(np.identity(3)-(V1@V1.T))@O1
                L2=(np.identity(3)-(V2@V2.T))@O2
                L=np.concatenate((L1,L2),axis=0)

                #calculate A vector
                A1 = (np.identity(3) - (V1 @ V1.T))
                A2 = (np.identity(3) - (V2 @ V2.T))
                A = np.concatenate((A1, A2), axis=0)

                N=A.T@A
                U=A.T@L

                X=np.linalg.inv(N)@U
                V=A@X-L

                #Calculate the accuresy of X
                Sx=(V.T@V)/(len(A)-len(A[0]))*np.linalg.inv(N)
                grdPointss[f"P{n}"]=[round(X[0],3),round(X[1],3),round(X[2],3)]
                grdPointss[f"RMS P{n}"]=[round(np.sqrt(Sx[0,0]),3),round(np.sqrt(Sx[1,1]),3),round(np.sqrt(Sx[2,2]),3)]

            if Method == "vector":
                #reshape for V
                V1=V1[:,0]
                V2 =V2[:, 0]

                #calculate A and L for adjusment
                A=np.array([[V1.T@V1,-V1.T@V2],[V1.T@V2,-V2.T@V2]])
                L=np.array([[(O2-O1).T@V1],[(O2-O1).T@V2]])
                Lambdaa=np.linalg.inv(A)@L

                F=O1+Lambdaa[0]*V1
                G=O2+Lambdaa[1]*V2

                X=(F+G)/2

                ##calculate the distance between the two rays
                d=np.linalg.norm(F-G)
                print(d)

                grdPointss[f"P{n}"] = [X[0], X[1], X[2]]
                grdPointss[f"RMS P{n}"] = None
        return grdPointss


    def ComputeDependentRelativeOrientation(self, imagePoints1, imagePoints2, initialValues):
        """
         Compute relative orientation parameters

        :param imagePoints1: points in the first image [m"m]
        :param imagePoints2: corresponding points in image 2(homology points) nx2 [m"m]
        :param initialValues: approximate values of relative orientation parameters

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type initialValues: np.array (6L,)

        :return: relative orientation parameters.

        :rtype: np.array 5x1 / ADD

        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])
            new = ImagePair(image1, image2)

            new.ComputeDependentRelativeOrientation(imagePoints1, imagePoints2, np.array([1, 0, 0, 0, 0, 0])))

        """
        pass  # delete after implementation


    def Build_A_B_W(self, cameraPoints1, cameraPoints2, x):
        """
        Function for computing the A and B matrices and vector w.
        :param cameraPoints1: points in the first camera system
        :param ImagePoints2: corresponding homology points in the second camera system
        :param x: initialValues vector by, bz, omega, phi, kappa ( bx=1)

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3
        :type x: np.array (5,1)

        :return: A ,B matrices, w vector

        :rtype: tuple
        """
        numPnts = cameraPoints1.shape[0] # Number of points

        dbdy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        dbdz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        dXdx = np.array([1, 0, 0])
        dXdy = np.array([0, 1, 0])


        # Compute rotation matrix and it's derivatives
        rotationMatrix2 = Compute3DRotationMatrix(x[2, 0], x[3, 0], x[4, 0])
        dRdOmega = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'omega')
        dRdPhi = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'phi')
        dRdKappa = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'kappa')

        # Create the skew matrix from the vector [bx, by, bz]
        bMatrix = ComputeSkewMatrixFromVector(np.array([1, x[0, 0], x[1, 0]]))

        # Compute A matrix; the coplanar derivatives with respect to the unknowns by, bz, omega, phi, kappa
        A = np.zeros((numPnts, 5))
        A[:, 0] = np.diag(
            np.dot(cameraPoints1, np.dot(dbdy, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to by
        A[:, 1] = np.diag(
            np.dot(cameraPoints1, np.dot(dbdz, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to bz
        A[:, 2] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdOmega, cameraPoints2.T))))  # derivative in respect to omega
        A[:, 3] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdPhi, cameraPoints2.T))))  # derivative in respect to phi
        A[:, 4] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdKappa, cameraPoints2.T))))  # derivative in respect to kappa

        # Compute B matrix; the coplanar derivatives in respect to the observations, x', y', x'', y''.
        B = np.zeros((numPnts, 4 * numPnts))
        k = 0
        for i in range(numPnts):
            p1vec = cameraPoints1[i, :]
            p2vec = cameraPoints2[i, :]
            B[i, k] = np.dot(dXdx, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 1] = np.dot(dXdy, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 2] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdx)
            B[i, k + 3] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdy)
            k += 4

        # w vector
        w = np.diag(np.dot(cameraPoints1, np.dot(bMatrix, np.dot(rotationMatrix2, cameraPoints2.T))))

        return A, B, w


    def ImagesToModel(self, imagePoints1, imagePoints2, Method):
        """
        Mapping points from image space to model space

        :param imagePoints1: points from the first image
        :param imagePoints2: points from the second image
        :param Method: method for intersection

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: corresponding model points
        :rtype: np.array nx3


        .. warning::

            This function is empty, need implementation

        .. note::

            One of the images is a reference, orientation of this image must be set.

        """


    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        pass  # delete after implementation


    def geometricIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on geometric calculations.

        :param cameraPoints1: points in the first image
        :param cameraPoints2: corresponding points in the second image

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3

        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """


    def vectorIntersction(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on vector calculations.

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx
        :type cameraPoints2: np.array nx


        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """


    def CollinearityIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray intersection based on the collinearity principle

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx2
        :type cameraPoints2: np.array nx2

        :return: corresponding ground points

        :rtype: np.array nx3

        .. warning::

            This function is empty, need implementation

        """


if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    leftCamPnts = np.array([[-4.83,7.80],
                            [-4.64, 134.86],
                            [5.39,-100.80],
                            [4.58,55.13],
                            [98.73,9.59],
                            [62.39,128.00],
                            [67.90,143.92],
                            [56.54,-85.76]])
    rightCamPnts = np.array([[-83.17,6.53],
                             [-102.32,146.36],
                             [-62.84,-102.87],
                             [-97.33,56.40],
                             [-3.51,14.86],
                             [-27.44,136.08],
                             [-23.70,152.90],
                             [-8.08,-78.07]])
    new = ImagePair(image1, image2)

    print(new.ComputeDependentRelativeOrientation(leftCamPnts, rightCamPnts, np.array([1, 0, 0, 0, 0, 0])))