import pandas as pd
from Point2D import Point2D
from math import inf


"""student_Name - Mor Levenhar and Tchelet Lev
Student_Id - 318301124, 206351611
Course Number - 014845
Course Name - introduction to computer mapping
Home Work number 4"""


class Route:  # This class calculates the shortest route between start and end points
    def __init__(self, vertices, edges, start_point, end_point):
        """vertices: dataframe of the x,y coordinate
        edges: dataframe which holding the number of start and end of each edge
        start_point: the number of the start point
        end_point: the number of the end point"""
        self.vertices = vertices
        self.edges = edges
        self.start = start_point
        self.final = end_point

    def dijkstra(self):
        """The function calculates the shortest route beetwen two points by using dijkstra algorithm.
        it returns list which contains the numbers of the points that the shortest route contains"""
        shortest_route = []
        point = self.start
        x_list = self.vertices["POINT_X"].values.tolist()  # Create a list with the x values
        y_list = self.vertices["POINT_Y"].values.tolist()  # Create a list with the y values
        Point_list = []
        checked = pd.DataFrame({"flag": [False] * len(x_list)})
        checked.to_numpy()
        

        point0_list = self.edges["POINT_0"].values.tolist()  # Create a list with the start points
        point1_list = self.edges["POINT_1"].values.tolist()  # Create a list with the end points
        matrix = [[inf for j in range(len(x_list))] for i in range(len(y_list))]
        for i in range(len(x_list)):  # Create Point2D list of the points
            p1 = Point2D(x_list[i], y_list[i],"")
            Point_list.append(p1)
        for i, val in enumerate(point0_list):  # Create adjacency matrix
            j = point1_list[i]
            matrix[val][j] = Point_list[val].distance_to_point(Point_list[j])
            matrix[j][val] = Point_list[val].distance_to_point(Point_list[j])

        distance = [inf] * len(x_list)  # List of distance from the origin point
        distance[self.start] = 0
        queue = [self.start]
        pi = [None] * len(x_list)  # List that holding the former point
        while len(queue) != 0:
            for j in range(len(x_list)):  # Fill in the pi list
                if matrix[queue[0]][j] != inf and not checked['flag'][j] and j not in queue:
                    queue.append(j)
                    pi[j] = queue[0]

            checked['flag'][self.start] = True
            min_val = inf
            min_index = 0
            curr = queue[0]
            """Check if the distance from the origin point + the distance to the current point to the next one
                            is less than the existing distance between the neighbor and the origin point, updates it
                            and mark the points that we checked"""
            for i, val in enumerate(distance):
                if distance[curr] + matrix[curr][i] <= distance[i] and matrix[curr][i] != inf and not checked['flag'][i] and \
                        pi[i] is not None:
                    distance[i] = distance[curr] + matrix[curr][i]
                    pi[i] = curr

            for i, val in enumerate(distance):  # Find the minimal distance and saves it in min_val
                if not checked['flag'][i]:
                    if min_val > val:
                        min_val = val
                        min_index = i
            if len(queue) == 1:
                min_index = queue[0]

            queue.remove(min_index)
            if len(queue) != 0:
                queue[0] = min_index
                '''point = queue[0]'''
            checked['flag'][min_index] = True

        father = pi[self.final]
        while father is not None:  # Create a list of the points we have to go through in the route
            shortest_route.append(father)
            father = pi[father]
        shortest_route = shortest_route[::-1]
        shortest_route.append(self.final)
        return shortest_route

