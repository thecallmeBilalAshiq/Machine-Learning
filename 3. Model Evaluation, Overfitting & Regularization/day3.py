import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Parameters
ACTIONS = ['up', 'down', 'left', 'right']
MOVES = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
GOAL_REWARD = 1
MOVE_PENALTY = -0.04
WALL_PENALTY = -1

EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.3

def load_maze(filename):
    with open(filename) as f:
        return [list(line.strip()) for line in f]

def get_start_goal(maze):
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == 'S':
                start = (r, c)
            if maze[r][c] == 'G':
                goal = (r, c)
    return start, goal

def step(state, action, maze):
    r, c = state
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]):
        if maze[nr][nc] == '#':
            return state, WALL_PENALTY
        elif maze[nr][nc] == 'G':
            return (nr, nc), GOAL_REWARD
        else:
            return (nr, nc), MOVE_PENALTY
    return state, WALL_PENALTY

def initialize_q(maze):
    Q = {}
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] != '#':
                Q[(r, c)] = {a: 0.0 for a in ACTIONS}
    return Q

def train_q(maze):
    Q = initialize_q(maze)
    start, goal = get_start_goal(maze)
    global EPSILON

    for ep in range(EPISODES):
        state = start
        steps = 0
        while state != goal and steps < 100:
            steps += 1
            if random.random() < EPSILON:
                action = random.choice(ACTIONS)
            else:
                action = max(Q[state], key=Q[state].get)

            next_state, reward = step(state, action, maze)
            if next_state not in Q:
                Q[next_state] = {a: 0.0 for a in ACTIONS}

            max_next = max(Q[next_state].values())
            Q[state][action] += ALPHA * (reward + GAMMA * max_next - Q[state][action])
            state = next_state

        EPSILON = max(0.01, EPSILON * 0.995)  # decay exploration

    return Q

def extract_best_path(Q, start, goal):
    path = []
    current = start
    visited = set()

    for _ in range(100):
        path.append(current)
        visited.add(current)
        if current == goal:
            break
        if current not in Q:
            break
        action = max(Q[current], key=Q[current].get)
        dr, dc = MOVES[action]
        next_cell = (current[0]+dr, current[1]+dc)
        if next_cell in visited:
            break
        current = next_cell
    return path

def render_policy(Q, maze, path, filename="final_policy.png"):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

    for r in range(len(maze)):
        for c in range(len(maze[0])):
            x, y = c, -r
            cell = maze[r][c]
            if cell == '#':
                ax.add_patch(patches.Rectangle((x, y), 1, 1, color='black'))
            elif cell == 'S':
                ax.add_patch(patches.Rectangle((x, y), 1, 1, color='blue'))
                ax.text(x + 0.5, y + 0.5, 'S', ha='center', va='center', color='white')
            elif cell == 'G':
                ax.add_patch(patches.Rectangle((x, y), 1, 1, color='green'))
                ax.text(x + 0.5, y + 0.5, 'G', ha='center', va='center', color='white')
            elif (r, c) in path:
                if (r, c) in Q:
                    best = max(Q[(r, c)], key=Q[(r, c)].get)
                    ax.text(x + 0.5, y + 0.5, arrows[best], ha='center', va='center', fontsize=12, color='red')

    ax.set_xlim(0, len(maze[0]))
    ax.set_ylim(-len(maze), 0)
    plt.axis('off')
    plt.savefig(filename)
    plt.show()
    print(f"✅ Saved to {filename}")

if __name__ == "__main__":
    maze = load_maze("maze_data.txt")
    Q = train_q(maze)
    start, goal = get_start_goal(maze)
    best_path = extract_best_path(Q, start, goal)
    render_policy(Q, maze, best_path)
