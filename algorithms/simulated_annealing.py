import numpy as np
import random
import math
from utils.distance_metrics import euclidean_distance

def total_distance(tour, coords):
    dist = 0.0
    for i in range(len(tour) - 1):
        dist += euclidean_distance(coords[tour[i]], coords[tour[i+1]])
    dist += euclidean_distance(coords[tour[-1]], coords[tour[0]])
    return dist

def simulated_annealing(coords, T_init=1000, T_min=1e-8, alpha=0.995, max_iter=1000):
    n = len(coords)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_distance = total_distance(current_tour, coords)
    best_tour = list(current_tour)
    best_distance = current_distance
    T = T_init

    while T > T_min:
        for _ in range(max_iter):
            i, j = sorted(random.sample(range(n), 2))
            new_tour = current_tour[:i] + current_tour[i:j+1][::-1] + current_tour[j+1:]
            new_distance = total_distance(new_tour, coords)
            delta = new_distance - current_distance

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_tour = new_tour
                current_distance = new_distance

                if current_distance < best_distance:
                    best_tour = list(current_tour)
                    best_distance = current_distance

        T *= alpha

    best_tour.append(best_tour[0])  # complete the tour
    return best_tour, best_distance
