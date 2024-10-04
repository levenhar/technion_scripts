import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyproj
import spectral
from scipy.interpolate import interp2d
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
# import open3d as o3d
from scipy.optimize import least_squares
import cv2
from scipy import stats



def byArea(n0, PointsArray):
    '''
    Mark the database type as byArea.
    The function creat the database matrix and calculate the value that define it.
    At least it close the small window and call to ploting function
    '''

    MaxValue = np.max(PointsArray, axis=0)
    MinValue = np.min(PointsArray, axis=0)

    Ly = MaxValue[1] - MinValue[1]
    Lx = MaxValue[0] - MinValue[0]
    # P = len(PointsArray) / n0
    # cellArea = (Lx * Ly) / P
    # a = (cellArea) ** 0.5
    a = 1 # pixel size in the spectarl Image
    n = len(PointsArray)

    Numx = int(np.ceil(Lx / a))
    Numy = int(np.ceil(Ly / a))
    CallMatrix = [[[] for i in range(Numx)] for j in range(Numy)]  # creating a matrix with empty lists.

    ny = np.floor((PointsArray[:, 1] - MinValue[1]) / a).astype(int)
    nx = np.floor((PointsArray[:, 0] - MinValue[0]) / a).astype(int)

    ny[ny < 0] = 0
    nx[nx < 0] = 0
    ny[ny > Numy] = Numy
    nx[nx > Numx] = Numx

    # x = list(map(lambda i: CallMatrix[ny[i]][nx[i]].append(i),range(n)))

    for i in range(len(PointsArray)):  # loop for place the indexes of the coordinate in the relevant cell.
        CallMatrix[ny[i]][nx[i]].append(PointsArray[i, :3])

    return CallMatrix


# data_frame = pd.read_csv("ground_points_Test.txt", delimiter=' ')
data_frame = pd.read_csv("ground_points.txt", delimiter=' ')
GroundPoints = data_frame.to_numpy()

AveragePointInCell = 10

Points = np.array(byArea(AveragePointInCell, GroundPoints[:, :3]))

#### create 2D array that each cell contine array of nx3 with all the points in the response cell ##################
Points = np.array([[np.array(Points[i, j]) for i in range(len(Points))] for j in range(len(Points[0]))])

# calculate the mean of each cell
PointsMean = np.array([[np.mean(Points[i, j], axis=0) for i in range(len(Points))] for j in range(len(Points[0]))]).T

# calculate the count pf point in each cell
densityMAP = np.array([[len(Points[i, j]) for i in range(len(Points))] for j in range(len(Points[0]))]).T

######## show box plot of points count in a cell ################################
# box = plt.boxplot(densityMAP.reshape(-1,1), patch_artist=True)
#
# box['boxes'][0].set(facecolor='blue', edgecolor='black', linewidth=2)
# box['medians'][0].set(color='black', linewidth=2)
# box['fliers'][0].set(marker='o', color='red', alpha=0.5)
#
# plt.title('Count of Points In a Cell')
# plt.show()
# #####################################################################################
#
# ##### display densityMAP ############################################################
# plt.imshow(densityMAP, cmap="gray")
# colorbar = plt.colorbar(label='Count of point in 2.5x2.5 meter cell')
# plt.show()
####################################################################################

##### Calculate l1 l2 l3 using covarince matrix -
DiffFromCenter = Points - PointsMean

AllMetrixs = []
ListPoints = []
SiglePoints = []  # list of cells (points) that have less than 3 points.
for i in range(len(Points)):
    for j in range(len(Points[0])):
        if len(DiffFromCenter[i, j]) == 0:  # empty cells
            continue
        # if (DiffFromCenter[i, j] == np.array([[0, 0, 0]])).all():
        elif len(DiffFromCenter[i, j]) < 3:
            SiglePoints.append(PointsMean[i, j])
            continue
        ExMul = np.einsum('ij,ik->ijk', DiffFromCenter[i, j],
                          DiffFromCenter[i, j])  # external multipication fot each row in the array
        AllMetrixs.append(np.average(ExMul, axis=0))  # calculate average for all the matrix in one cell
        ListPoints.append(PointsMean[i, j])

AllMetrixs = np.array(AllMetrixs)
SubsampleCloud = np.array(ListPoints)
SiglePoints = np.array(SiglePoints)

### Convert from ITM to UTM 36N  for adjust with the spectral image#####################################################
itm = pyproj.Proj(
    '+proj=tmerc +lat_0=31.73439361111111 +lon_0=35.20451694444445 +k=1.0000067 +x_0=219529.584 +y_0=626907.39 +ellps=GRS80 +towgs84=-48,55,52,0,0,0,0 +units=m +no_defs')
utm36 = pyproj.Proj('+proj=utm +zone=36 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

itm_easting = SubsampleCloud[:, 0]
itm_northing = SubsampleCloud[:, 1]

utm_easting, utm_northing = pyproj.transform(itm, utm36, itm_easting, itm_northing)

SubsampleCloud = np.vstack((utm_easting, utm_northing, SubsampleCloud[:, 2])).T
#######################################################################################################################

eigenvalues, eigenvectors = np.linalg.eigh(AllMetrixs)


####### displayt the normal of each cell #################################################################################
# Create or load a point cloud

# points = SubsampleCloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
#
# # Estimate normals
# #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#
# # Create an array of points where the normals originate (same as point cloud points)
# origins = np.asarray(pcd.points)
#
# # Calculate the endpoints of the normals
# normals = eigenvectors[:,:,0]
# endpoints = origins + 0.05 * normals  # The scale factor of 0.05 is for visualization
#
# # Combine origins and endpoints for visualization
# all_points = np.vstack((origins, endpoints))
#
# # Create line segments between each point and its corresponding endpoint
# lines = [[i, i + len(origins)] for i in range(len(origins))]
#
# # Create a LineSet for the normals
# line_set = o3d.geometry.LineSet(
#     points=o3d.utility.Vector3dVector(all_points),
#     lines=o3d.utility.Vector2iVector(lines),
# )
#
# # Visualize the point cloud and the normals
# o3d.visualization.draw_geometries([pcd, line_set])


###### calculate feture for optimization #################################################

V3 = eigenvectors[:, 0]

l1 = eigenvalues[:, 2]
l2 = eigenvalues[:, 1]
l3 = eigenvalues[:, 0]

L = (l1 - l2) / (l1 + 10 ** -5)
P = (l2 - l3) / (l1 + 10 ** -5)
S = l3 / (l1 + 10 ** -5)
O = np.cbrt(l1 * l2 * l3)

A = (l1 - l3) / (l1 + 10 ** -5)

lnn1 = l1 * np.log(l1 + 10 ** -5)
lnn2 = l2 * np.log(l2 + 10 ** -5)
lnn3 = l3 * np.log(l3 + 10 ** -5)
E = -(lnn1 + lnn2 + lnn3)

summ = l1 + l2 + l3
C = l3 / (summ + 10 ** -5)

FetureArray = np.vstack((L, P, S, O, A, summ, C)).T

########Reading the Spectral Image#################################
hdr_file_path = 'VE_VM01_VSC_L2VALD_ISRAW902_20200930_CLIP.hdr'

# Read the ENVI file along with its header
ImgHeader = spectral.open_image(hdr_file_path)

# Access the actual image data
SpectralImage = ImgHeader.load()
SpectralImage = np.array(SpectralImage)

# plt.imshow(SpectralImage[:, :, [7, 3, 1]] / np.max(SpectralImage[:, :, [7, 3, 1]]))
# plt.show()
##################################################################


MapInfo = ImgHeader.metadata["map info"]
x0 = int(MapInfo[3])
y0 = int(MapInfo[4])
celSize = int(MapInfo[5])

X0 = np.array([[x0], [y0]])

x = np.array(range(x0, x0 + celSize * int(ImgHeader.ncols), 5))
y = np.array(range(y0, y0 - celSize * int(ImgHeader.nrows), -5))

# Create mesh grids from coordinates
y_mesh, x_mesh = np.meshgrid(y, x, indexing='ij')
x_coords_reshaped = x_mesh.reshape(-1)
y_coords_reshaped = y_mesh.reshape(-1)

SpectralValue = []
for b in range(int(ImgHeader.nbands)):
    data_reshaped = SpectralImage[:, :, b].reshape(-1)

    interpolated_values = griddata((x_coords_reshaped, y_coords_reshaped), data_reshaped, SubsampleCloud[:, :2],
                                   method='linear')
    SpectralValue.append(interpolated_values)

SpectralArray = np.array(SpectralValue).T
COLORDCLOUD = np.concatenate((SubsampleCloud, SpectralValue[7].reshape(-1, 1) / np.max(SpectralValue[7]),
                              SpectralValue[3].reshape(-1, 1) / np.max(SpectralValue[3]),
                              SpectralValue[1].reshape(-1, 1) / np.max(SpectralValue[1])), axis=1)
FullCloud = np.concatenate((SubsampleCloud, eigenvalues, FetureArray, SpectralArray), axis=1)
np.savetxt('COLORDCLOUD.csv', COLORDCLOUD, delimiter=',', fmt='%f')
np.savetxt('FullCloud.csv', FullCloud, delimiter=',', fmt='%f')

######  careate DTM ##############################################################################################

MaxValue = np.max(GroundPoints[:,:2], axis=0)
MinValue = np.min(GroundPoints[:,:2], axis=0)
celSize = 0.2

x = np.array(range(int(MinValue[0])+6, int(np.ceil(MaxValue[0]))-6))
y = np.array(range(int(np.ceil(MaxValue[1]))-6, int(MinValue[1])+6,-1))

# Create mesh grids from coordinates
y_mesh, x_mesh = np.meshgrid(y, x, indexing='ij')

DTM = griddata(GroundPoints[:,:2], GroundPoints[:, 2],(x_mesh, y_mesh), method='linear')

# plt.imshow(DTM, cmap = "gray")
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()

def Hillshade(Z, AZg, cellsize, DTM):
    Hdx = 1 / (8 * cellsize) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Hdy = Hdx.T
    dldx = cv2.filter2D(DTM, -1, Hdx)
    dldy = cv2.filter2D(DTM, -1, Hdy)
    #######################################################################################
    AZm = 2 * np.pi - (AZg + np.pi / 2)
    S = np.sqrt(dldx ** 2 + dldy ** 2)

    Asp = np.arctan2(dldy, dldx)
    HS = 255 * (np.cos(Z) * np.cos(S) + np.sin(Z) * np.sin(S) * np.cos(AZm - Asp))
    return HS

Z = np.deg2rad(45)
AZg = np.deg2rad(135)

# Hillshade1 = Hillshade(Z, AZg, 1, DTM)
# plt.xticks([])
# plt.yticks([])
# plt.imshow(Hillshade1, cmap = "gray")
# plt.show()


#### calculate derivatives ##################################################################################

cellsize = 1

Hdx = 1 / (2 * cellsize) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Hdy = Hdx.T
dzdx = cv2.filter2D(DTM, -1, Hdx)
dzdy = cv2.filter2D(DTM, -1, Hdy)

Hdxx = 1/(cellsize**2)*np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
Hdyy = Hdxx.T

dzdxx = cv2.filter2D(DTM, -1, Hdxx)
dzdyy = cv2.filter2D(DTM, -1, Hdyy)

Hdxy = 1/(4*cellsize**2)*np.array([[-1, 0, 1], [0, 0, 0], [-1, 0, 1]])
dzdxy = cv2.filter2D(DTM, -1, Hdxy)
gradianteSIZE = np.sqrt(dzdx**2 + dzdy**2)


t = 0.001
c=0
# t = 0.185

AllMapsC = []
while t <= 0.35:
    print(c)
    Gequal0 = np.greater(gradianteSIZE, t)
    #########################################
    #calculate l1 l2 l3 using DTM

    ClasterMAP = np.zeros(DTM.shape)
    eigenvaluesLIST = []
    gradsizeLIST = []

    for i in range(len(DTM)):
        for j in range(len(DTM[0])):

            H = np.array([[dzdxx[i,j], dzdxy[i,j]],[dzdxy[i,j] , dzdyy[i,j]]])

            grad = np.array([[dzdx[i,j]], [dzdy[i,j]]]).astype(np.float)

            eigenvalues, eigenvectors = np.linalg.eig(H)

            if eigenvalues[0] < eigenvalues[1]:
                eigenvalues[0], eigenvalues[1] = eigenvalues[1], eigenvalues[0]

            eigenvaluesLIST.append(eigenvalues.reshape(1,-1))
            gradsizeLIST.append(grad.reshape(1,-1))
            V1 = eigenvectors[:, -1]
            l1 = eigenvalues[0]
            l2 = eigenvalues[1]

            if not Gequal0[i,j]:   #the gradiante is equal to zero
                if l1 < -t and l2 < -t:
                    ### if true the point on a pic (1)
                    ClasterMAP[i,j] = 1
                    continue
                elif l1 > t and l2 > t:
                    ### if true the point on a pit (2)
                    ClasterMAP[i,j] = 2
                    continue
            else:
                if abs(l1) < t and l2 < -t:
                    ### if true the point on a ridge (3)
                    ClasterMAP[i,j] = 3
                    continue
                elif l1 > t and abs(l2) < t:
                    ### if true the point on a valley (4)
                    ClasterMAP[i,j] = 4
                    continue
                # if l1 < -t and abs((V1.reshape(1, -1) @ grad)[0, 0]) < t:
                #     ### if true the point on a ridge (3)
                #     ClasterMAP[i,j] = 3
                #     continue
                # elif l1 > t and abs((V1.reshape(1, -1) @ grad)[0, 0]) < t:
                #     ### if true the point on a valley (4)
                #     ClasterMAP[i,j] = 4
                #     continue

            if l1 * l2 < -t and abs(l1)>t and abs(l2)>t:
                ### if true the point on a saddle (5)
                ClasterMAP[i,j] = 5
                continue
            elif abs(l1) < t and abs(l2) < t:
                ### if true the point on a flat (6)
                ClasterMAP[i,j] = 6
                continue
    AllMapsC.append(ClasterMAP[:, :, None])
    t += 0.005
    c+=1
# ClasterMAp111 = cv2.applyColorMap(ClasterMAP.astype(np.uint8), cv2.COLORMAP_JET)
# cv2.imwrite(f'Maps\CategorizedArea_t={t}.jpg', ClasterMAp111)

ClasterMAP = np.concatenate(AllMapsC,axis=2)
ClasterMAP = stats.mode(ClasterMAP, axis=2)
ClasterMAP = np.squeeze(ClasterMAP.mode)


plt.imshow(ClasterMAP, cmap = "jet")
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

hist_data = ClasterMAP.flatten()
plt.hist(hist_data,bins=50, alpha=0.7)
plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# map = ax1.imshow(ClasterMAP, cmap = "jet")
# plt.colorbar(map, ax=ax1)
# ax1.set_xticks([])
# ax1.set_yticks([])
#
# hist_data = ClasterMAP.flatten()
# ax2.hist(hist_data,bins=50, alpha=0.7)

# fig.savefig(f'CategorizedArea_t={t}.jpg')
# plt.show()




###### calculate l1 l2 l3 using bi-quadratic surface #############################################################

# Function to fit
def bi_quadratic(params, x, y, z, w):
    a, b, c, d, e, f = params
    return w * (e * x ** 2 + f * y ** 2 + d * x * y + b * x + c * y + a - z)


def fit_surface_to_cells(cells):
    fitted_params = {}
    Traslation = []
    rows, cols = cells.shape
    for i in range(rows):
        for j in range(cols):
            if len(cells[i, j]) == 0:
                continue
            # Collect points from the cell itself
            cell_points = cells[i, j]
            x_cell, y_cell, z_cell = cell_points[:, 0], cell_points[:, 1], cell_points[:, 2]
            w_cell = np.ones(z_cell.shape)

            # Collect points from neighbors
            x_neigh, y_neigh, z_neigh, w_neigh = [], [], [], []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if 0 <= i + dx < rows and 0 <= j + dy < cols:
                        neigh_points = cells[i + dx, j + dy]
                        if len(neigh_points) <= 0:
                            continue
                        x_neigh.extend(neigh_points[:, 0])
                        y_neigh.extend(neigh_points[:, 1])
                        z_neigh.extend(neigh_points[:, 2])
                        w_neigh.extend(0.5 * np.ones(neigh_points[:, 0].shape))
            if len(x_cell) + len(x_neigh) <= 6:  # if there is less than 6 points we cant calculate the surface
                continue
            # Combine cell and neighbor points
            x = np.concatenate([x_cell, x_neigh])
            y = np.concatenate([y_cell, y_neigh])
            z = np.concatenate([z_cell, z_neigh])
            w = np.concatenate([w_cell, w_neigh])

            Mx = np.average(x)
            My = np.average(y)
            Mz = np.average(z)

            Traslation.append([Mx, My, Mz])

            x -= Mx
            y -= My
            z -= Mz

            # A1 = np.ones((len(x),1))
            # A2 = x.reshape(-1,1)
            # A3 = y.reshape(-1,1)
            # A4 = z.reshape(-1,1)
            # A5 = (x*y).reshape(-1,1)
            # A6 = (x**2).reshape(-1,1)
            # A7 = (y**2).reshape(-1,1)
            #
            # A = np.concatenate((A1,A2,A3,A4,A5,A6,A7),axis=1)
            #
            # P = np.diag(w)
            # N = A.T@P@A
            #
            # eigval, eigvec = np.linalg.eig(N)
            #
            # index = np.where(np.abs(eigval) == np.min(np.abs(eigval)))
            #
            # X = eigvec[:,index[0]]
            # X = X / -X[3]
            # X = X[[0,1,2,4,5,6]]

            # Fit bi-quadratic surface
            initial_guess = [1, 1, 1, 1, 1, 1]
            res = least_squares(bi_quadratic, initial_guess, args=(x, y, z, w))
            fitted_params[(i, j)] = res.x

            # fitted_params[(i, j)] = X.reshape(-1)

    return fitted_params, np.array(Traslation)


# Fit bi-quadratic surface to each cell
fitted_params, TraslationOUT = fit_surface_to_cells(Points)

t = 1e-4
LableList = []
Xlist = []
Ylist = []
Zlist = []


eigenvaluesLIST = []
gradLIST = []
SIZElist = []
c=0
for P in fitted_params:
    parm = fitted_params[P]
    H = np.array([[parm[4], 0.5 * parm[3]], [0.5 * parm[3], parm[5]]])

    X = PointsMean[P[0], P[1]][0]
    Y = PointsMean[P[0], P[1]][1]
    Z = PointsMean[P[0], P[1]][2]

    Xlist.append(X)
    Ylist.append(Y)
    Zlist.append(Z)


    X -= TraslationOUT[c, 0]
    Y -= TraslationOUT[c, 1]
    Z -= TraslationOUT[c, 2]

    dx = parm[1] + parm[3] * Y + 2 * parm[4] * X
    dy = parm[2] + parm[3] * X + 2 * parm[5] * Y
    grad = np.array([[dx], [dy]]).astype(np.float)
    gradSize = np.sqrt(grad[0,0] ** 2 + grad[1,0] ** 2)

    gradLIST.append(grad)
    SIZElist.append(gradSize)

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    eigenvaluesLIST.append(eigenvalues.reshape(1,-1))

    V1 = eigenvectors[:, -1]

    l1 = eigenvalues[1]
    l2 = eigenvalues[0]

    if gradSize < t:   #the gradiante is equal to zero
        if l1 < -t and l2 < -t:
            ### if true the point on a pic (2)
            LableList.append(2)
            continue
        elif l1 > t and l2 > t:
            ### if true the point on a pit (3)
            LableList.append(3)
            continue
        elif abs(l1) < t and l2 < -t:
            ### if true the point on a ridge (4)
            LableList.append(4)
            continue
        elif l1 > t and abs(l2) < t:
            ### if true the point on a valley (5)
            LableList.append(5)
            continue
    else:
        if l1 < -t and abs((V1.reshape(1, -1) @ grad)[0, 0]) < t:
            ### if true the point on a ridge (4)
            LableList.append(4)
            continue
        elif l1 > t and abs((V1.reshape(1, -1) @ grad)[0, 0]) < t:
            ### if true the point on a valley (5)
            LableList.append(5)
            continue

    if l1 * l2 < -t and abs(l1)>t and abs(l2)>t:
        ### if true the point on a saddle (6)
        LableList.append(6)
        continue
    elif abs(l1) < t and abs(l2) < t:
        ### if true the point on a flat (7)
        LableList.append(7)
        continue
    else:
        LableList.append(0)


eigenvaluesLIST = np.concatenate(eigenvaluesLIST)
gradLIST = np.concatenate(gradLIST)
SIZElist = np.array(SIZElist)

x=0


ClassfiedPoints = np.concatenate((np.array(Xlist).reshape(-1, 1), np.array(Ylist).reshape(-1, 1), np.array(Zlist).reshape(-1, 1), np.array(LableList).reshape(-1, 1)), axis=1)
np.savetxt('ClassfiedPoints0.5m.csv', ClassfiedPoints, delimiter=',', fmt='%f')
