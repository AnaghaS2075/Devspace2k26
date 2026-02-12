import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

GRID_SIZE = 30
lambda_risk = 8
K = 20

# Create grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

np.random.seed(42)
for _ in range(100):
    x = np.random.randint(0, GRID_SIZE)
    y = np.random.randint(0, GRID_SIZE)
    grid[x, y] = 1

start = (0, 0)
goal = (25, 25)
grid[start] = 0
grid[goal] = 0


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal, risk_map):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        neighbors = [
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1),
        ]

        for neighbor in neighbors:
            if (
                0 <= neighbor[0] < rows
                and 0 <= neighbor[1] < cols
                and grid[neighbor] == 0
            ):
                tentative_g = (
                    g_score[current]
                    + 1
                    + lambda_risk * risk_map[neighbor]
                )

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None


# Initial positions
drone_pos = start
intruder = [12, 12]

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

for step in range(60):

    # ---------------------------
    # Step 1: Plan path first
    # ---------------------------
    risk_map = np.zeros((GRID_SIZE, GRID_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            distance = np.sqrt((x - intruder[0])**2 + (y - intruder[1])**2)
            risk_map[x, y] = K / (1 + distance)

    path = astar(grid, drone_pos, goal, risk_map)

    # ---------------------------
    # Step 2: Move intruder toward future path
    # ---------------------------
    if path and len(path) > 3:
        target = path[3]
    else:
        target = drone_pos

    if intruder[0] < target[0]:
        intruder[0] += 1
    elif intruder[0] > target[0]:
        intruder[0] -= 1

    if intruder[1] < target[1]:
        intruder[1] += 1
    elif intruder[1] > target[1]:
        intruder[1] -= 1

    intruder[0] = np.clip(intruder[0], 0, GRID_SIZE - 1)
    intruder[1] = np.clip(intruder[1], 0, GRID_SIZE - 1)

    # ---------------------------
    # Step 3: Recompute risk map after intruder moved
    # ---------------------------
    risk_map = np.zeros((GRID_SIZE, GRID_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            distance = np.sqrt((x - intruder[0])**2 + (y - intruder[1])**2)
            risk_map[x, y] = K / (1 + distance)

    path = astar(grid, drone_pos, goal, risk_map)

    # ---------------------------
    # Step 4: Move drone one step
    # ---------------------------
    if path and len(path) > 1:
        drone_pos = path[1]


    # Visualization
    ax.clear()
    ax.imshow(grid, cmap="gray_r")
    ax.imshow(risk_map, cmap="Reds", alpha=0.4)

    if path:
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py)

    ax.scatter(goal[1], goal[0], marker="x")
    ax.scatter(drone_pos[1], drone_pos[0], marker="o")
    ax.scatter(intruder[1], intruder[0], marker="s")

    ax.set_title("Phase 3: Dynamic Replanning")
    plt.pause(0.2)

plt.ioff()
plt.show()
