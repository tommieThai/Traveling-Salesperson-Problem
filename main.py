from utils.tsp_loader import load_tsp_file
from algorithms.nearest_neighbor import nearest_neighbor
from algorithms.simulated_annealing import simulated_annealing
import matplotlib.pyplot as plt
import time


def plot_tour(coords, tour, title):
    x = [coords[i][0] for i in tour]
    y = [coords[i][1] for i in tour]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def main():
    datasets = ['att48.tsp', 'wi29.tsp', 'dj38.tsp']

    for name in datasets:
        filepath = f'data/{name}'
        print(f"\nRunning algorithms on {name}...")

        coords = load_tsp_file(filepath)

        # Nearest Neighbor
        start = time.time()
        tour_nn, dist_nn = nearest_neighbor(coords)
        end = time.time()
        print(f"[Nearest Neighbor] Distance: {dist_nn:.2f} | Time: {end - start:.4f}s")
        plot_tour(coords, tour_nn, f"Nearest Neighbor Tour - {name}")

        # Simulated Annealing
        start = time.time()
        tour_sa, dist_sa = simulated_annealing(coords)
        end = time.time()
        print(f"[Simulated Annealing] Distance: {dist_sa:.2f} | Time: {end - start:.4f}s")
        plot_tour(coords, tour_sa, f"Simulated Annealing Tour - {name}")


if __name__ == "__main__":
    main()
