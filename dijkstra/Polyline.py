from Point2D import Point2D
from Segment import Segment

class Polyline:
    def __init__(self, points, name=""):
        '''
        :param points: points list, from this points the segment will be created.
        :type points: list
        :param name: an ID
        :type name: string
        '''
        self.Name=name
        self.Lines=points

    @property
    def Name(self):
        return self._name
    @Name.setter
    def Name(self,name):
        self._name = name

    @property
    def Lines(self):
        return self._lines
    @Lines.setter
    def Lines(self, points):
        self._lines =  list(map(lambda p,x:Segment(p,x),points[:-1],points[1:]))


    def __getitem__(self, item):
        return self._lines[item]

    def poly_length(self):
        '''
        calculate the polyline length
        :return:the length
        :rtype:float
        '''
        lengths = []
        for line in self._lines:
            lengths.append(line.line_length())
        return sum(lengths)

