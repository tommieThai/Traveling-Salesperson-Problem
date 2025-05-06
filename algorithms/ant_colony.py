import numpy as np
import random
from utils.distance_metrics import euclidean_distance

def total_distance(tour, coords):
    dist = 0.0
    for i in range(len(tour) - 1):
        dist += euclidean_distance(coords[tour[i]], coords[tour[i+1]])
    dist += euclidean_distance(coords[tour[-1]], coords[tour[0]])
    return dist

def ant_colony(coords, n_ants=20, n_best=5, n_iterations=100, decay=0.95, alpha=1, beta=2):
    n = len(coords)
    dist_matrix = [[euclidean_distance(coords[i], coords[j]) for j in range(n)] for i in range(n)]
    pheromone = [[1 / (n * n) for _ in range(n)] for _ in range(n)]

    def pick_move(pheromone, dist, visited):
        pheromone = np.array(pheromone)
        dist = np.array(dist)
        row = pheromone ** alpha * ((1.0 / (dist + 1e-10)) ** beta)
        row[list(visited)] = 0
        norm_row = row / row.sum()
        move = np.random.choice(range(n), 1, p=norm_row)[0]
        return move

    best_dist = float('inf')
    best_tour = []

    for iteration in range(n_iterations):
        all_tours = []
        for ant in range(n_ants):
            tour = [random.randint(0, n - 1)]
            visited = set(tour)
            for _ in range(n - 1):
                move = pick_move(pheromone[tour[-1]], dist_matrix[tour[-1]], visited)
                tour.append(move)
                visited.add(move)
            all_tours.append((tour, total_distance(tour, coords)))

        all_tours.sort(key=lambda x: x[1])
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= decay

        for tour, dist in all_tours[:n_best]:
            for i in range(n):
                pheromone[tour[i]][tour[(i + 1) % n]] += 1.0 / dist

        if all_tours[0][1] < best_dist:
            best_dist = all_tours[0][1]
            best_tour = all_tours[0][0]

    best_tour.append(best_tour[0])
    return best_tour, best_dist
