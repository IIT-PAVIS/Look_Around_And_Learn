import numpy as np

class AStarGraph():

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.open_nodes = []
        self.path = []
        self.g_score = [0]*len(nodes)
        self.f_score = [0]*len(nodes)

    def dist(self, a, b):
        aa = self.nodes[a]
        bb = self.nodes[b]
        return np.sqrt( (aa[0]-bb[0])**2 + (aa[1]-bb[1])**2 )

    def get_f_score(self, n):
        return self.f_score[n]

    def set_f_score(self, n, f):
        self.f_score[n] = f

    def set_g_score(self, n, g):
        self.g_score[n] = g

    def get_g_score(self, n):
        return self.g_score[n]

    def set_start_node(self, n):
        self.start_node = n

    def set_end_node(self, n):
        self.end_node = n

    def set_path(self, path):
        self.path = path

    def find_neighbours(self, index):
        neighbors = []
        for e in self.edges:
            if e[0] == index:
                neighbors.append(e[1])
            if e[1] == index:
                neighbors.append(e[0])

        return list(set(neighbors))


def astar(grid):
    open_nodes = [grid.start_node]
    closed_nodes = []
    came_from = {}

    grid.set_g_score(grid.start_node, 0)
    grid.set_f_score(grid.start_node, grid.get_g_score(grid.start_node) + grid.dist(grid.start_node,grid.end_node) )

    # main loop
    while len(open_nodes)>0:
        minf = 100000.0
        current = []
        for n in open_nodes:
            if grid.get_f_score(n) < minf:
                current = n
                minf = grid.get_f_score(n)

        if current == grid.end_node:
            grid.path = reconstruct_path(came_from, current)            
            break

        open_nodes.remove(current)
        closed_nodes.append(current)

        nn = grid.find_neighbours(current)
        for n in nn:
            if n in closed_nodes:
                continue
            tentative_g_score = grid.get_g_score(current) + grid.dist(current,n)

            if n not in open_nodes or tentative_g_score < grid.get_g_score(n):
                came_from[n] = current
                grid.set_g_score(n, tentative_g_score)
                grid.set_f_score(n, grid.get_g_score(n) + grid.dist(n,grid.end_node) )
                if n not in open_nodes:
                    open_nodes.append(n)

        grid.open_nodes = open_nodes


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path



