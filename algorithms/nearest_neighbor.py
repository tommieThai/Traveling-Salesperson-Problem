import numpy as np
from utils.distance_metrics import euclidean_distance


def nearest_neighbor(coords):
    n = len(coords)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    total_distance = 0.0

    for _ in range(1, n):
        last = tour[-1]
        nearest = -1
        min_dist = float('inf')
        for i in range(n):
            if not visited[i]:
                dist = euclidean_distance(coords[last], coords[i])
                if dist < min_dist:
                    min_dist = dist
                    nearest = i
        tour.append(nearest)
        visited[nearest] = True
        total_distance += min_dist

    # return to start
    total_distance += euclidean_distance(coords[tour[-1]], coords[tour[0]])
    tour.append(tour[0])

    return tour, total_distance
