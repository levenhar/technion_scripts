import numpy as np
from ImagePair import ImagePair
from SingleImage import SingleImage
from Camera import Camera

class ImageTriple(object):
    def __init__(self, imagePair1, imagePair2):
        """
        Inisialize the ImageTriple class

        :param imagePair1: first image pair
        :param imagePair2: second image pair

        .. warning::

            Check if the relative orientation is solved for each image pair
        """
        if imagePair1.image1.isSolved and imagePair1.image2.isSolved and imagePair2.image2.isSolved:
            self.__imagePair1 = imagePair1
            self.__imagePair2 = imagePair2
        else:
            print("The Inner Orientation of the images isn't solved")

    @property
    def imagePair1(self):

        return self.__imagePair1

    @property
    def imagePair2(self):

        return self.__imagePair2


    def ComputeScaleBetweenModels(self,cameraPoint1, cameraPoint2, cameraPoint3):
        """
         Compute scale between two models given the relative orientation

         :param cameraPoints1: camera point in first camera space
         :param cameraPoints2: camera point in second camera space
         :param cameraPoints3:  camera point in third camera space

         :type cameraPoints1: np.array 1x3
         :type cameraPoints2: np.array 1x3
         :type cameraPoints3: np.array 1x3


         .. warning::

             This function is empty, need implementation
        """

        v1b=np.array([[cameraPoint1[0,0]],[cameraPoint1[1,0]],[-self.__imagePair1.image1.camera.focalLength]])
        v2b = np.array([[cameraPoint2[0,0]], [cameraPoint2[1,0]], [-self.__imagePair1.image1.camera.focalLength]])
        v3b = np.array([[cameraPoint3[0,0]], [cameraPoint3[1,0]], [-self.__imagePair1.image1.camera.focalLength]])

        I=np.eye(3)
        R1=np.eye(3)
        R2=self.__imagePair1.RotationMatrix_Image2
        R3=self.__imagePair1.RotationMatrix_Image2@self.__imagePair2.RotationMatrix_Image2

        V1=R1@v1b
        V2=R2@v2b
        V3=R3@v3b

        O1=np.array([[0],[0],[0]])
        O2=self.__imagePair1.PerspectiveCenter_Image2.reshape(3,1)
        O3=self.__imagePair2.PerspectiveCenter_Image2.reshape(3,1)

        d1=np.cross(V1.reshape((3)),V2.reshape((3))).reshape((3,1))
        d2=np.cross(V2.reshape((3)),V3.reshape((3))).reshape((3,1))

        AA1=np.concatenate((V1,d1,-V2),axis=1)
        K2=(np.linalg.inv(AA1)@O2)[2,0]

        AA2=np.concatenate((O3,V3,-d2),axis=1)
        scale=(np.linalg.inv(AA2)@(K2*V2))[0,0]


        return scale

    def RayIntersection(self, cameraPoints1, cameraPoints2, cameraPoints3):
        """
        Compute coordinates of the corresponding model point by the geomatric technique

        :param cameraPoints1: points in camera1 coordinate system
        :param cameraPoints2: points in camera2 coordinate system
        :param cameraPoints3: points in camera3 coordinate system

        :type cameraPoints1 np.array nx3
        :type cameraPoints2: np.array nx3
        :type cameraPoints3: np.array nx3

        :return: point in model coordinate system
        :rtype: np.array nx3

        .. warning::

            This function is empty' need implementation
        """
        scales = []
        for i in range(11):
            P1 = np.array([[cameraPoints1[i, 0]], [cameraPoints1[i, 1]]])
            P2 = np.array([[cameraPoints2[i, 0]], [cameraPoints2[i, 1]]])
            P3 = np.array([[cameraPoints3[i, 0]], [cameraPoints3[i, 1]]])
            S = self.ComputeScaleBetweenModels(P1, P2, P3)
            scales.append(S)

        Avg_Scale = np.average(scales)

        grdPointss = np.array([0,0,0]).reshape((1,3))

        # store the rotation Matrix of ech camera
        R1 = self.imagePair1.RotationMatrix_Image1
        R2 = self.imagePair1.RotationMatrix_Image2
        R3 = self.imagePair1.RotationMatrix_Image2 @ self.imagePair1.RotationMatrix_Image2

        b32 = self.imagePair2.PerspectiveCenter_Image2

        # store the prespective center of each image
        O1 = self.imagePair1.PerspectiveCenter_Image1
        O2 = self.imagePair1.PerspectiveCenter_Image2
        O3 = O2+Avg_Scale*R2@b32




        for n in range(len(cameraPoints1)):
            V1 = R1 @ np.array([cameraPoints1[n, 0], cameraPoints1[n, 1], -self.imagePair1.image1.camera.focalLength]).T
            V2 = R2 @ np.array([cameraPoints2[n, 0], cameraPoints2[n, 1], -self.imagePair1.image2.camera.focalLength]).T
            V3 = R3 @ np.array([cameraPoints3[n, 0], cameraPoints3[n, 1], -self.imagePair2.image2.camera.focalLength]).T
            V1 = np.reshape(V1, (3, 1))
            V2 = np.reshape(V2, (3, 1))
            V3 = np.reshape(V3, (3, 1))

            # normalized the vectors
            V1 = V1 / np.linalg.norm(V1)
            V2 = V2 / np.linalg.norm(V2)
            V3 = V3 / np.linalg.norm(V3)


            # calculate L vector
            L1 = (np.identity(3) - (V1 @ V1.T)) @ O1
            L2 = (np.identity(3) - (V2 @ V2.T)) @ O2
            L3 = (np.identity(3) - (V3 @ V3.T)) @ O3
            L = np.concatenate((L1, L2, L3), axis=0)

            # calculate A vector
            A1 = (np.identity(3) - (V1 @ V1.T))
            A2 = (np.identity(3) - (V2 @ V2.T))
            A3 = (np.identity(3) - (V3 @ V3.T))
            A = np.concatenate((A1, A2, A3), axis=0)

            N = A.T @ A
            U = A.T @ L

            X = (np.linalg.inv(N) @ U).reshape((3,1))
            '''V = A @ X - L

            # Calculate the accuresy of X
            Sx = (V.T @ V) / (len(A) - len(A[0])) * np.linalg.inv(N)
            grdPointss[f"P{n}"] = [round(X[0], 3), round(X[1], 3), round(X[2], 3)]
            grdPointss[f"RMS P{n}"] = [round(np.sqrt(Sx[0, 0]), 3), round(np.sqrt(Sx[1, 1]), 3),
                                       round(np.sqrt(Sx[2, 2]), 3)]'''

            grdPointss=np.concatenate((grdPointss,X.T),axis=0)
        return grdPointss[1:,:]


    def drawModles(self, imagePair1, imagePair2, modelPoints1, modelPoints2):
        """
        Draw two models in the same figure

        :param imagePair1: first image pair
        :param imagePair2:second image pair
        :param modelPoints1: points in the firt model
        :param modelPoints2:points in the second model

        :type modelPoints1: np.array nx3
        :type modelPoints2: np.array nx3

        :return: None

        .. warning::
            This function is empty, need implementation
        """
        imagePair1.drawImagePair(modelPoints1)
        imagePair2.drawImagePair(modelPoints2)
if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    image3 = SingleImage(camera)
    imagePair1 = ImagePair(image1, image2)
    imagePair2 = ImagePair(image2, image3)
    imageTriple1 = ImageTriple(imagePair1, imagePair2)
