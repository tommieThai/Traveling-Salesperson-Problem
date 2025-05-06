import random
import numpy as np
from deap import base, creator, tools, algorithms
from utils.distance_metrics import euclidean_distance

def total_distance(tour, coords):
    dist = 0.0
    for i in range(len(tour) - 1):
        dist += euclidean_distance(coords[tour[i]], coords[tour[i+1]])
    dist += euclidean_distance(coords[tour[-1]], coords[tour[0]])
    return dist

# Only create DEAP classes if not already created
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def genetic_algorithm(coords, population_size=100, generations=300, cx_pb=0.7, mut_pb=0.2):
    n = len(coords)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_tour(ind):
        return (total_distance(ind, coords),)

    toolbox.register("evaluate", eval_tour)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=cx_pb, mutpb=mut_pb, ngen=generations, stats=None, halloffame=hof, verbose=False)

    best_tour = hof[0] + [hof[0][0]]
    best_distance = total_distance(hof[0], coords)
    return best_tour, best_distance
