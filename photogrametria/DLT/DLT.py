import numpy as np
from scipy import linalg
import MatrixMethods as MM


class DLT():
    @staticmethod
    def dltCalculation(GCP, ImagePoints):
        GCP = np.concatenate((GCP, np.ones((len(GCP), 1))), axis=1)
        A = np.zeros((2 * len(GCP), 12))
        for i in range(len(GCP)):
            A11 = A22 = np.zeros((1, 4))
            A12 = (-GCP[i, :]).reshape((1, 4))
            A13 = (ImagePoints[i, 1] * GCP[i, :]).reshape((1, 4))
            A1 = np.concatenate((A11, A12, A13), axis=1)

            A23 = (-ImagePoints[i, 0] * GCP[i, :]).reshape((1, 4))
            A2 = np.concatenate((-A12, A22, A23), axis=1)

            Ai = np.concatenate((A1, A2))
            A[2 * i:2 * i + 2, :] = Ai

        N = A.T @ A

        eigenvalues, eigenvectors = np.linalg.eig(N)
        lowest_eigenvalue_index = np.argmin(eigenvalues)
        lowest_eigenvector = eigenvectors[:, lowest_eigenvalue_index]

        print(np.linalg.cond(N))

        return lowest_eigenvector

    @staticmethod
    def Pmatrix2KR(p):
        P = p.reshape(3, 4)
        c = -np.linalg.inv(P[:3,:3])@(P[:,3].reshape((3,1)))
        K, R = linalg.rq(P[:3,:3])
        if np.sign(K[2, 2]) == 1:
            K = -1*K

        o = np.array([[-1 * np.sign(K[0, 0]), 0, 0],
                      [0, -1 * np.sign(K[1, 1]), 0],
                      [0, 0, 1 * np.sign(K[2, 2])]])

        K = (1 / abs(K[2, 2])) * K@o
        R = o@R

        return c, R, K

    @staticmethod
    def unittest(Image1, ControlPoints,ImagePoints):
        '''
        function that get syntetic data without noise an chacke if the DLT work
        :param Image1: the Image we solve.
        :type Image1: Image Object
        :param ControlPoints: Points in the world system
        :type ControlPoints: np. array
        :param ImagePoints: Points in the image system
        :type ImagePoints: np. array
        :return: true if the DLT work, False otherwise
        :rtype: bool
        '''

        P = DLT.dltCalculation(ControlPoints, ImagePoints)
        X0, R, K = DLT.Pmatrix2KR(P)
        O, F, KK = MM.ExtractRotationAngles(R.T,"Rad")

        ##### calculate the diffrences between the truw value and the calculated one.
        X0t = np.array([[Image1.exteriorOrientationParameters["X0"]], [Image1.exteriorOrientationParameters["Y0"]], [Image1.exteriorOrientationParameters["Z0"]]])
        dx0 = np.abs(X0t-X0)
        Doritation = np.array([[O-Image1.exteriorOrientationParameters["omega"]],[F-Image1.exteriorOrientationParameters["phi"]], [KK - Image1.exteriorOrientationParameters["kappa"]]])
        Kt = Image1.camera.Calibration_Matrix
        dK = np.abs(K-Kt)

        print(f"Position error - {dx0} [m]")
        print(f"Orientation error - {Doritation * 206265} ['']")
        print("Calibration error [pix]")
        MM.PrintMatrix(dK)
        if max(dx0)>10**-4 or max(Doritation*206265)>1 or max(dK.reshape(-1,))>10**-3:
            return False
        return True