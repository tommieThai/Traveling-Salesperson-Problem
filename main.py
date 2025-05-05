from utils.tsp_loader import load_tsp_file
from algorithms.nearest_neighbor import nearest_neighbor
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
    datasets = ['att48.tsp', 'fri26.tsp', 'gr17.tsp']

    for name in datasets:
        filepath = f'data/{name}'
        print(f"\nRunning Nearest Neighbor on {name}...")

        coords = load_tsp_file(filepath)
        start = time.time()
        tour, total_distance = nearest_neighbor(coords)
        end = time.time()

        print(f"Total Distance: {total_distance:.2f}")
        print(f"Execution Time: {end - start:.4f} seconds")

        plot_tour(coords, tour, f"Nearest Neighbor Tour - {name}")


if __name__ == "__main__":
    main()
