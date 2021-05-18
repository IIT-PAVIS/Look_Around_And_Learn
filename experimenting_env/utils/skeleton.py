# type: ignore
import copy

import cv2
import numpy as np

from experimenting_env.utils.astar import *

startNode, endNode = None, None


def mouseCallback(action, x, y, flags, *userdata):
    global startNode, endNode
    if action == cv2.EVENT_LBUTTONDOWN:
        startNode = [x, y]
    if action == cv2.EVENT_MBUTTONDOWN:
        endNode = [x, y]


def concave(vertices, index):
    currVert = vertices[index]
    nextVert = vertices[(index + 1) % len(vertices)]
    prevVert = vertices[-1]
    if index != 0:
        prevVert = vertices[index - 1]

    left = [currVert[0] - prevVert[0], currVert[1] - prevVert[1]]
    right = [nextVert[0] - currVert[0], nextVert[1] - currVert[1]]

    cross = (left[0] * right[1]) - (left[1] * right[0])

    return cross < 0


def segmentsCrossing(a, b, c, d):
    denominator = ((b[0] - a[0]) * (d[1] - c[1])) - ((b[1] - a[1]) * (d[0] - c[0]))
    if denominator == 0:
        return False

    numerator1 = ((a[1] - c[1]) * (d[0] - c[0])) - ((a[0] - c[0]) * (d[1] - c[1]))
    numerator2 = ((a[1] - c[1]) * (b[0] - a[0])) - ((a[0] - c[0]) * (b[1] - a[1]))

    if numerator1 == 0 or numerator2 == 0:
        return False

    r = numerator1 / denominator
    s = numerator2 / denominator

    return (r > 0 and r < 1) and (s > 0 and s < 1)


def inside(vertices, position, toleranceOnOutside=True):
    point = position
    epsilon = 0.5
    inside = False

    if len(vertices) < 3:
        return False

    oldPoint = vertices[-1]
    oldSqDist = np.sum(np.square(oldPoint - point))

    for i in range(len(vertices)):
        newPoint = vertices[i]
        newSqDist = np.sum(np.square(newPoint - point))

        if (
            oldSqDist
            + newSqDist
            + 2.0 * np.sqrt(oldSqDist * newSqDist)
            - np.sum(np.square(newPoint - oldPoint))
            < epsilon
        ):
            return toleranceOnOutside

        if newPoint[0] > oldPoint[0]:
            left = oldPoint
            right = newPoint
        else:
            left = newPoint
            right = oldPoint

        if (
            left[0] < point[0]
            and point[0] <= right[0]
            and (point[1] - left[1]) * (right[0] - left[0])
            < (right[1] - left[1]) * (point[0] - left[0])
        ):
            inside = not inside

        oldPoint = newPoint
        oldSqDist = newSqDist

    return inside


def polygonInside(verices1, vertices2):
    for v in vertices2:
        if not inside(verices1, v, False):
            return False
    return True


def inLineOfSight(polygons, start, end, epsilon=5):
    isInside = False
    for vertices in polygons:
        if inside(vertices, start) and inside(vertices, end):
            isInside = True
    if not isInside:
        return False

    # if np.linalg.norm(end-start) < epsilon:
    #     return True

    for vertices in polygons:
        n = len(vertices)
        for i in range(n):
            if segmentsCrossing(start, end, vertices[i], vertices[(i + 1) % n]):
                return False

    return True  # inside(vertices, (start + end) / 2.0)


# returns indexes
def neighbors(graph, index):
    neighbors = []
    edges = graph[1]
    for e in edges:
        if e[0] == index:
            neighbors.append(e[1])
        if e[1] == index:
            neighbors.append(e[0])

    return list(set(neighbors))


def plan(graph, walkable):
    nodes = graph[0]
    # edges = graph[1]

    lastIndex = len(nodes) - 1

    # if inLineOfSight(walkable, nodes[0], nodes[lastIndex]):
    #     trajectory.append(lastIndex)
    #     return trajectory

    fwdTrajectory = []
    fwdTrajectory.append(0)
    fwdTrajLen = 0
    index = 0
    while index != len(nodes) - 1:
        if inLineOfSight(walkable, nodes[index], nodes[-1]):
            fwdTrajectory.append(lastIndex)
            break  # return trajectory
        nn = neighbors(graph, index)
        distToGoal = 1e7
        farthestNN = None
        dist = 0
        for n in nn:
            if n == len(nodes) - 1:
                fwdTrajectory.append(lastIndex)
                break  # return trajectory
            dist = np.linalg.norm(nodes[n] - nodes[-1])
            if dist < distToGoal:
                farthestNN = n
                distToGoal = dist
        fwdTrajectory.append(farthestNN)
        fwdTrajLen += dist
        index = farthestNN

    bwdTrajectory = []
    bwdTrajectory.append(lastIndex)
    bwdTrajLen = 0
    index = lastIndex
    while index > 0:
        if inLineOfSight(walkable, nodes[index], nodes[0]):
            bwdTrajectory.append(0)
            break  # return bwdTrajectory
        nn = neighbors(graph, index)
        distToStart = 1e7
        farthestNN = None
        dist = 0
        for n in nn:
            if n == 0:
                bwdTrajectory.append(0)
                break  # return bwdTrajectory
            dist = np.linalg.norm(nodes[n] - nodes[0])
            if dist < distToStart:
                farthestNN = n
                distToStart = dist
        bwdTrajectory.append(farthestNN)
        bwdTrajLen += dist
        index = farthestNN

    print("traj len", fwdTrajLen, bwdTrajLen)
    if fwdTrajLen <= bwdTrajLen:
        return fwdTrajectory
    else:
        return bwdTrajectory

    return fwdTrajectory


def find_skeleton(img, threshold=127):
    skeleton = np.zeros(img.shape, np.uint8)
    eroded = np.zeros(img.shape, np.uint8)
    temp = np.zeros(img.shape, np.uint8)

    _, thresh = cv2.threshold(img, threshold, 255, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    iters = 0
    while True:
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh  # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton, iters)


def nodesFromImage(
    img, step=2, minDist=20, minWallDist=4
):  # minDist=40, minWallDist=10):
    skel, _ = find_skeleton(img)
    nodes = []
    for i in range(0, skel.shape[0], step):
        for j in range(0, skel.shape[1], step):
            if skel[i, j] == 255:
                tentativeNode = [j, i]  # in opencv coords

                doInsert = True
                # remove points close to the walls
                for x in range(
                    max(i - minWallDist, 0), min(i + minWallDist, img.shape[0] - 1)
                ):
                    for y in range(
                        max(j - minWallDist, 0), min(j + minWallDist, img.shape[1] - 1)
                    ):
                        if x != y:
                            if img[x, y] == 0:
                                doInsert = False
                # remove points close to each other
                for node in nodes:
                    if (
                        np.linalg.norm(np.array(tentativeNode) - np.array(node))
                        < minDist
                    ):
                        doInsert = False
                if doInsert:
                    nodes.append([j, i])  # in opencv coords
    return nodes, skel


def closest_node_to_position(position, nodes):
    min_dist = 1000000.0
    min_index = 0
    nodes.append([position[0], position[1]])
    for i, node in enumerate(nodes[:-1]):
        aa, bb = node, nodes[-1]
        dist = np.sqrt((aa[0] - bb[0]) ** 2 + (aa[1] - bb[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            min_index = i

    nodes.pop(-1)
    return min_index


def checkVisibleFromImage(img, a, b):
    def lerp(points, steps):
        divisor = steps - 1
        if divisor == 0:
            return points[0]

        dx = (points[1][0] - points[0][0]) / divisor
        dy = (points[1][1] - points[0][1]) / divisor
        x, y = points[0]
        for _ in range(steps):
            yield x, y
            x += dx
            y += dy

    for x, y in lerp([a, b], int(np.linalg.norm(np.array(a) - np.array(b)))):
        if img[int(y), int(x)] == 0:
            return False
    return True


def edgesFromImage(img, nodes):
    edges = []
    for i, node in enumerate(nodes):
        for j, othernode in enumerate(nodes):
            if checkVisibleFromImage(img, node, othernode):
                edges.append([i, j])
    return edges


def mouseCallback(action, x, y, flags, *userdata):
    global startNode, endNode
    if action == cv2.EVENT_LBUTTONDOWN:
        startNode = [x, y]
    if action == cv2.EVENT_MBUTTONDOWN:
        endNode = [x, y]


def do_plan(
    bwimg, startNode, endNode, random_goal=False, closest_goal=True, position=[0, 0]
):

    img = cv2.cvtColor(bwimg, cv2.COLOR_GRAY2BGR)
    curr_img = img.copy()

    nodes, skel = nodesFromImage(bwimg)

    nodes.insert(0, startNode)
    if not (random_goal or closest_goal):

        nodes.append(endNode)
    edges = edgesFromImage(bwimg, nodes)
    for edge in edges:
        cv2.line(
            curr_img, tuple(nodes[edge[0]]), tuple(nodes[edge[1]]), (150, 150, 255), 1
        )
    for node in nodes:
        cv2.circle(curr_img, tuple(node), 6, (0, 0, 255), -1)
    graph = AStarGraph(nodes, edges)

    graph.set_start_node(0)
    if random_goal:
        graph.set_end_node(np.random.randint(1, len(nodes) - 1))
    elif closest_goal:
        graph.set_end_node(closest_node_to_position(position, nodes))

    else:
        graph.set_end_node(len(nodes) - 1)
    astar(graph)
    path = graph.path
    goals = []
    for index in path:
        goals.append(nodes[index])
        cv2.circle(curr_img, tuple(nodes[index]), 10, (255, 100, 100), 2)
    for i in range(len(path)):

        if i > 0:
            cv2.line(
                curr_img,
                tuple(nodes[path[i - 1]]),
                tuple(nodes[path[i]]),
                (255, 100, 100),
                2,
            )
    cv2.circle(curr_img, tuple(nodes[0]), 10, (0, 255, 0), 2)
    cv2.circle(curr_img, tuple(nodes[graph.end_node]), 10, (0, 0, 255), 2)

    return goals, curr_img
