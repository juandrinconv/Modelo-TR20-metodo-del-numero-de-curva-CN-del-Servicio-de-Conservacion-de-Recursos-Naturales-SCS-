
import copy


dataset = [
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9,  10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
    [16, 17, 18, 19, 20],
    [11, 12, 13, 14, 15],
    [6,  7,  8,  9,  10],
    [1,  2,  3,  4,  5],
]


class Node(object):

    def __init__(self, elevation):
        self.elevation = elevation
        self.node_ds = None
        self.magnitude = 0


class Grid(object):
    def __init__(self, data):
        self.data = data

    @property
    def row_max(self):
        return len(self.data) - 1

    @property
    def col_max(self):
        return len(self.data[0]) - 1

    def get_nodes(self):
        nodes = []
        for line in self.data:
            row = []
            for elevation in line:
                node = Node(elevation)
                row.append(node)
            nodes.append(row)
        return nodes

    def get_surrounding(self, i, j):
        nodes = []
        nodes_in = self.get_nodes()
        if 0 < i < self.row_max:  # middle rows
            if 0 < j < self.row_min:  # middle rows
            pass
        elif 0 < i:  # last row
            pass
        else:  # first row




    def get_magnitudes(self):
        nodes = self.get_nodes()
        magnitudes = []
        for line in nodes:
            row = []
            for node in line:
                nodes_in = copy.copy(nodes)
                trigger = True
                while trigger:
                    trigger = False
                    for i, n in enumerate(nodes_in):
                        if node == node_curr:
                            links_out.append(links_in.pop(i))
                            node_curr = link.node_2
                            trigger = True
                row.append(node)
            nodes.append(row)
        return nodes