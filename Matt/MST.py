from queue import PriorityQueue
from math import inf


def calDis(x1, x2, y1, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def minCostConnectPoints(points: list[list[int]]) -> int:
    # convert points to dictionary
    Q = PriorityQueue()
    Q.put((0, 0))
    pred = {}
    V = set()  # processed vertices
    E = {}  # dict of MST edges and lengths
    d = {i: inf for i in range(len(points))}
    d[0] = 0

    # Build the MST by iterating through nodes in Q
    while not Q.empty():
        (_, i) = Q.get()  # pop the node with the smallest cost
        print(i)
        p = (points[i][0], points[i][1])
        if i in V: continue  # v was already done, go again
        V.add(i)
        if i in pred:
            k = points[pred[i]]
            E[(i, pred[i])] = calDis(k[0], points[i][0], k[1], points[i][1])
        for j in range(len(points)):
            if j not in V and i != j:
                new_dist = calDis(points[j][0], points[i][0], points[j][1], points[i][1])
                if new_dist < d[j]:
                    pred[j] = i
                    d[j] = new_dist
                    Q.put((d[j], j))

    return sum(x for x in E.values())


print(minCostConnectPoints([[3,12],[-2,5],[-4,1]]))