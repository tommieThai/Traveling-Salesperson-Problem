from utils.tsp_loader import load_tsp_file
from algorithms.nearest_neighbor import nearest_neighbor
import matplotlib.pyplot as plt


def plot_tour(coords, tour):
    x = [coords[i][0] for i in tour]
    y = [coords[i][1] for i in tour]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-')
    plt.title("TSP Tour - Nearest Neighbor")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def main():
    # Replace with your own .tsp file path
    filepath = 'data/berlin52.tsp'
    coords = load_tsp_file(filepath)

    tour, total_distance = nearest_neighbor(coords)
    print("Tour:", tour)
    print("Total Distance:", total_distance)

    plot_tour(coords, tour)


if __name__ == "__main__":
    main()
