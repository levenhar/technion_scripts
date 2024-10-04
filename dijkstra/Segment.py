import Point2D

class Segment:
    def __init__(self,start,end,name =""):
        '''
        :param start:start point
        :type start: Point2D
        :param end: end point
        :type end: Point2D
        :param name: an ID
        :type name: string
        '''
        self.Start = start
        self.End = end
        self._name = name

    @property
    def Start(self):
        return self._start
    @Start.setter
    def Start(self,start):
        self._start=start


    @property
    def End(self):
        return self._end
    @End.setter
    def End(self,end):
        self._end = end

    def __repr__(self):
        return f"Name={self._name}, Start Point: " + str(self._start) + ", End Point: " + str(self._end)

    def line_length(self):
        '''
        calculate the segment length
        :return: the length
        :rtype: float
        '''
        return self._start.distance_to_point(self._end)

