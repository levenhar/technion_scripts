from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
import numpy as np
import PhotoViewer as PV
import matplotlib.pyplot as plt


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
    def image1(self):

        return self.__image1

    @property
    def image2(self):

        return self.__image2

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
        return Compute3DRotationMatrix(self.__relativeOrientationImage1[3], self.__relativeOrientationImage1[4],
                                       self.__relativeOrientationImage1[5])

    @property
    def RotationMatrix_Image2(self):
        """
        return the rotation matrix of the second image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage2[3], self.__relativeOrientationImage2[4],
                                       self.__relativeOrientationImage2[5])

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
        # Add -f as z coordinate for all the points
        C1 = -self.__image1.camera.focalLength * np.ones((len(imagePoints1), 1))
        imagePoints1 = np.concatenate((imagePoints1, C1), axis=1)
        C2 = -self.__image2.camera.focalLength * np.ones((len(imagePoints1), 1))
        imagePoints2 = np.concatenate((imagePoints2, C2), axis=1)

        dx=np.inf
        j=0

        while np.linalg.norm(dx)>10**-3:
            A, B, w = self.Build_A_B_W(imagePoints1,imagePoints2,initialValues)
            M=B@B.T
            N=A.T@np.linalg.inv(M)@A
            u=A.T@np.linalg.inv(M)@w
            dx=-np.linalg.inv(N)@u
            v = -B.T @ np.linalg.inv(M) @(A@dx+w)
            dx = np.reshape(dx, (5, 1))
            initialValues+=dx
            k=0
            for i in range(len(imagePoints1)):
                imagePoints1[i, 0] += v[k]
                imagePoints1[i, 1] += v[k + 1]
                imagePoints2[i, 0] += v[k + 2]
                imagePoints2[i, 1] += v[k + 3]
                k+=4
            j+=1
        print(j)

        sigma=np.sqrt(v.T@v/(np.size(B,0)-5))
        initialValues=np.insert(initialValues,0,1)
        print(initialValues)
        self.__relativeOrientationImage2=initialValues
        self.__isSolved = True
        initialValues[3:]=initialValues[3:]
        Sx=sigma**2*np.linalg.inv(N)

        return initialValues, v, sigma, Sx, np.linalg.inv(N)


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

    def ModelTransformation(self, modelPoints, scale):
        """
        Transform model from the current coordinate system to other coordinate system

        :param modelPoints: coordinates in current model space
        :param scale: scale between the two coordinate systems

        :type modelPoints: np.array nx3
        :type scale: float

        :return: corresponding coordinates in the other coordinate system

        :rtype: np.array nx3

        .. warning::

            This function is empty, needs implementation

        """

        ScaleMatrix=scale*np.eye(3)
        RealPoints=ScaleMatrix@modelPoints.T

        return RealPoints.T

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
        grdPointss = {}
        # store the prespective center of each image
        O1 = self.PerspectiveCenter_Image1
        O2 = self.PerspectiveCenter_Image2
        # store the rotation Matrix of ech camera
        R1 = self.RotationMatrix_Image1
        R2 = self.RotationMatrix_Image2
        for n in range(len(cameraPoints1)):
            V1 = R1 @ np.array([cameraPoints1[n, 0], cameraPoints1[n, 1], -self.__image1.camera.focalLength]).T
            V2 = R2 @ np.array([cameraPoints2[n, 0], cameraPoints2[n, 1], -self.__image2.camera.focalLength]).T
            V1 = np.reshape(V1, (3, 1))
            V2 = np.reshape(V2, (3, 1))

            # normalized the vectors
            V1 = V1 / np.linalg.norm(V1)
            V2 = V2 / np.linalg.norm(V2)

            # calculate L vector
            L1 = (np.identity(3) - (V1 @ V1.T)) @ O1
            L2 = (np.identity(3) - (V2 @ V2.T)) @ O2
            L = np.concatenate((L1, L2), axis=0)

            # calculate A vector
            A1 = (np.identity(3) - (V1 @ V1.T))
            A2 = (np.identity(3) - (V2 @ V2.T))
            A = np.concatenate((A1, A2), axis=0)

            N = A.T @ A
            U = A.T @ L

            X = np.linalg.inv(N) @ U
            V = A @ X - L

            # Calculate the accuresy of X
            Sx = (V.T @ V) / (len(A) - len(A[0])) * np.linalg.inv(N)
            grdPointss[f"P{n}"] = [round(X[0], 3), round(X[1], 3), round(X[2], 3)]
            grdPointss[f"RMS P{n}"] = [round(np.sqrt(Sx[0, 0]), 3), round(np.sqrt(Sx[1, 1]), 3),
                                       round(np.sqrt(Sx[2, 2]), 3)]
        return grdPointss

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
        grdPointss = {}
        # store the prespective center of each image
        O1 = self.PerspectiveCenter_Image1
        O2 = self.PerspectiveCenter_Image2
        # store the rotation Matrix of ech camera
        R1 = self.RotationMatrix_Image1
        R2 = self.RotationMatrix_Image2
        for n in range(len(cameraPoints1)):
            V1 = R1 @ np.array([cameraPoints1[n, 0], cameraPoints1[n, 1], -self.__image1.camera.focalLength]).T
            V2 = R2 @ np.array([cameraPoints2[n, 0], cameraPoints2[n, 1], -self.__image2.camera.focalLength]).T
            V1 = np.reshape(V1, (3, 1))
            V2 = np.reshape(V2, (3, 1))

            # reshape for V
            V1 = V1[:, 0]
            V2 = V2[:, 0]

            # calculate A and L for adjusment
            A = np.array([[V1.T @ V1, -V1.T @ V2], [V1.T @ V2, -V2.T @ V2]])
            L = np.array([[(O2 - O1).T @ V1], [(O2 - O1).T @ V2]])
            Lambdaa = np.linalg.inv(A) @ L

            F = O1 + Lambdaa[0] * V1
            G = O2 + Lambdaa[1] * V2

            X = (F + G) / 2

            grdPointss[f"P{n}"] = [X[0], X[1], X[2]]
            grdPointss[f"RMS P{n}"] = None
        return grdPointss

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

    def RotationLevelModel(self, constrain1, constrain2):
        """
        Compute rotation matrix from the current model coordinate system to the other coordinate system

        :param constrain1: constrain of the first axis
        :param constrain2: constrain of the second axis

        :type constrain1: tuple
        :type constrain2: tuple

        :return: rotation matrix

        :rtype: np.array 3x3

        .. note::

            The vector data included in the two constrains must be normalized

            The two constrains should be given to two different axises, if not return identity matrix

        """
        if constrain1[0]==constrain2[0]:
            return np.eye(3)

        V1=constrain1[1]/np.linalg.norm(constrain1[1])
        V2=constrain2[1]/np.linalg.norm(constrain2[1])

        if constrain1[0]=="x" and constrain2[0]=="y":
            X = V1.reshape((3,1))
            Y = V2.reshape((3,1))
            Z = np.cross(X.reshape(3),Y.reshape(3)).reshape((3,1))
        elif constrain1[0]=="y" and constrain2[0]=="x":
            X = V2.reshape((3,1))
            Y = V1.reshape((3,1))
            Z = np.cross(X.reshape(3),Y.reshape(3)).reshape((3,1))
        elif constrain1[0]=="x" and constrain2[0]=="z":
            X = V1.reshape((3,1))
            Z = V2.reshape((3,1))
            Y = np.cross(Z.reshape(3),X.reshape(3)).reshape((3,1))
        elif constrain1[0]=="z" and constrain2[0]=="x":
            Z = V1.reshape((3,1))
            X = V2.reshape((3,1))
            Y = np.cross(Z.reshape(3),X.reshape(3)).reshape((3,1))
        elif constrain1[0] == "z" and constrain2[0] == "y":
            Z = V1.reshape((3,1))
            Y = V2.reshape((3,1))
            X = np.cross(Y.reshape(3), Z.reshape(3)).reshape((3,1))
        elif constrain1[0] == "y" and constrain2[0] == "z":
            Y = V1.reshape((3,1))
            Z = V2.reshape((3,1))
            X = np.cross(Y.reshape(3), Z.reshape(3)).reshape((3,1))
        else:
            print("input error - the axis does not define well")
            return

        R=np.concatenate((X,Y,Z),axis=1)
        return R.T

    def drawImagePair(self,ModelPoints):
        '''
        this function draw the points and the cameras in 3D axis according to the model coordinate system
        :param ModelPoints: points in model system
        :type ModelPoints: array nx3
        :return: None
        :rtype:
        '''
        imageWidth = 5472
        ImageHeight = 3648
        f=self.__image2.camera.focalLength


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(ModelPoints)):
            ax.text(ModelPoints[i,0],ModelPoints[i,1],ModelPoints[i,2],f"{i+1}")
        #Plot the Model points
        ax.scatter(ModelPoints[:,0],ModelPoints[:,1],ModelPoints[:,2],'red')

        #Plot the Model edges
        ax.plot(ModelPoints[3:6,0],ModelPoints[3:6,1],ModelPoints[3:6,2],"blue")
        ax.plot(ModelPoints[:3, 0], ModelPoints[:3, 1], ModelPoints[:3, 2],"blue")
        ax.plot(ModelPoints[6:10, 0], ModelPoints[6:10, 1], ModelPoints[6:10, 2],"blue")
        ax.plot([ModelPoints[7, 0],ModelPoints[4, 0]], [ModelPoints[7, 1],ModelPoints[4, 1]], [ModelPoints[7, 2],ModelPoints[4, 2]],"blue")
        ax.plot([ModelPoints[8, 0], ModelPoints[5, 0]], [ModelPoints[8, 1], ModelPoints[5, 1]],[ModelPoints[8, 2], ModelPoints[5, 2]],"blue")
        ax.plot([ModelPoints[9, 0], ModelPoints[6, 0]], [ModelPoints[9, 1], ModelPoints[6, 1]],[ModelPoints[9, 2], ModelPoints[6, 2]],"blue")

        #plot camera 1
        x0 = np.reshape(self.PerspectiveCenter_Image1, (3, 1))
        PV.drawOrientation(self.RotationMatrix_Image1, x0, 0.3,ax)  # Image 1
        PV.drawImageFrame(imageWidth, ImageHeight, self.RotationMatrix_Image1, x0, f,0.0001,ax)
        PV.drawRays(ModelPoints, x0,ax,"c")

        # plot camera 2
        x0 = np.reshape(self.PerspectiveCenter_Image2, (3, 1))
        PV.drawOrientation(self.RotationMatrix_Image2, x0, 0.3,ax)  # Image 2
        PV.drawImageFrame(imageWidth, ImageHeight, self.RotationMatrix_Image2, x0, f,0.0001,ax)
        PV.drawRays(ModelPoints, x0,ax,"m")

        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')


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

    IP=np.array([ 0, 0, 0, 0, 0], dtype="float64")
    IP=np.reshape(IP,(5,1))

    print(new.ComputeDependentRelativeOrientation(leftCamPnts, rightCamPnts,IP ))