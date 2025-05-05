def load_tsp_file(filepath):
    coords = []
    with open(filepath, 'r') as file:
        start = False
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                start = True
                continue
            if line == "EOF":
                break
            if start:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
    return coords
