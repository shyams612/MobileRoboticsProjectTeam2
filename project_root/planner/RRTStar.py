# ============================================
# RRT* PLANNER
# ============================================
from dataclasses import dataclass
from typing import Optional, List, Tuple
from project_root.environment.RandomEnvironment import RandomEnvironment
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass

@dataclass
class Node:
    x: float
    y: float
    parent: Optional[int]
    cost: float     # Cost from root to this node


class RRTStar:
    def __init__(self,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 env: RandomEnvironment,
                 step_size: float = 1.5,
                 goal_radius: float = 2.0,
                 max_iters: int = 2000,
                 rewire_radius: float = 4.0):

        self.start = start
        self.goal = goal
        self.env = env
        
        self.step_size = step_size
        self.goal_radius = goal_radius
        # Probability of sampling the exact goal (goal bias)
        self.goal_sample_rate = 0.05
        self.max_iters = max_iters
        self.rewire_radius = rewire_radius
        
        # Node list
        self.nodes: List[Node] = [
            Node(start[0], start[1], parent=None, cost=0.0)
        ]
        
        # Path will be stored after search
        self.final_path = None
        self.all_edges = []  # For visualization

    # --------------------------
    # Utility
    # --------------------------

    def distance(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def sample_point(self):
        # With a small probability, sample the exact goal to bias the tree
        if random.random() < self.goal_sample_rate:
            # Only sample the goal if the goal cell is free
            if self.env.is_free(self.goal[0], self.goal[1]):
                return (self.goal[0], self.goal[1])
        return self.env.sample_free_point()

    def nearest_node_index(self, point):
        dists = [(node.x - point[0])**2 + (node.y - point[1])**2
                 for node in self.nodes]
        return int(np.argmin(dists))

    def steer(self, from_node: Node, to_point):
        """
        Return a new node that is STEP_SIZE distance toward target point.
        """
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = np.hypot(dx, dy)

        if dist <= self.step_size:
            new_x, new_y = to_point
        else:
            new_x = from_node.x + (dx / dist) * self.step_size
            new_y = from_node.y + (dy / dist) * self.step_size

        return new_x, new_y

    def get_neighbors(self, new_node):
        """All nodes within rewire radius"""
        neighbors = []
        for i, node in enumerate(self.nodes):
            if self.distance((node.x, node.y), (new_node.x, new_node.y)) <= self.rewire_radius:
                neighbors.append(i)
        return neighbors

    # --------------------------
    # Main search
    # --------------------------

    def search(self):
        for it in range(self.max_iters):

            # 1. Sample random point
            rand_point = self.sample_point()

            # 2. Find nearest node
            nearest_idx = self.nearest_node_index(rand_point)
            nearest_node = self.nodes[nearest_idx]

            # 3. Steer toward sampled point
            new_x, new_y = self.steer(nearest_node, rand_point)

            # Check collision
            if not self.env.is_free(new_x, new_y):
                continue
            if not self.env.is_collision_free((nearest_node.x, nearest_node.y),
                                              (new_x, new_y)):
                continue

            new_node = Node(new_x, new_y, parent=nearest_idx,
                            cost=nearest_node.cost + self.distance((nearest_node.x, nearest_node.y),
                                                                   (new_x, new_y)))

            # 4. Choose optimal parent among neighbors
            neighbors = self.get_neighbors(new_node)
            best_parent = nearest_idx
            best_cost = new_node.cost

            for i in neighbors:
                n = self.nodes[i]
                if self.env.is_collision_free((n.x, n.y), (new_x, new_y)):
                    temp_cost = n.cost + self.distance((n.x, n.y), (new_x, new_y))
                    if temp_cost < best_cost:
                        best_parent = i
                        best_cost = temp_cost

            new_node.parent = best_parent
            new_node.cost = best_cost

            # Add node
            new_index = len(self.nodes)
            self.nodes.append(new_node)
            
            # Store edge for visualization
            self.all_edges.append(((self.nodes[best_parent].x, self.nodes[best_parent].y),
                                   (new_x, new_y)))

            # 5. Rewire
            for i in neighbors:
                if i == best_parent:
                    continue

                n = self.nodes[i]
                new_cost = new_node.cost + self.distance((new_node.x, new_node.y),
                                                         (n.x, n.y))

                if new_cost < n.cost:
                    if self.env.is_collision_free((new_node.x, new_node.y), (n.x, n.y)):
                        # Update parent
                        self.nodes[i].parent = new_index
                        self.nodes[i].cost = new_cost

            # 6. Check if goal reached (within goal region)
            if self.distance((new_x, new_y), self.goal) < self.goal_radius:
                # Prefer snapping to the exact goal point when possible
                # (i.e., goal cell is free and the segment to goal is collision-free).
                try_connect_goal = False
                if self.env.is_free(self.goal[0], self.goal[1]):
                    if self.env.is_collision_free((new_x, new_y), self.goal):
                        try_connect_goal = True

                if try_connect_goal:
                    goal_node = Node(self.goal[0], self.goal[1], parent=new_index,
                                     cost=new_node.cost + self.distance((new_x, new_y), self.goal))
                    goal_index = len(self.nodes)
                    self.nodes.append(goal_node)
                    # store final edge for visualization
                    self.all_edges.append(((new_x, new_y), (self.goal[0], self.goal[1])))
                    print(f"Goal reached and snapped to exact goal at iteration {it}")
                    return self.reconstruct_path(goal_index)
                else:
                    # If exact goal cannot be connected due to collision, still return
                    # the path to the node inside the goal region (existing behavior).
                    print(f"Goal region reached at iteration {it} (goal not directly connectable)")
                    return self.reconstruct_path(new_index)

        # Final attempt: try to connect any existing node to the exact goal
        best_idx = None
        best_dist = float('inf')
        for i, n in enumerate(self.nodes):
            d = self.distance((n.x, n.y), self.goal)
            if d < best_dist and self.env.is_collision_free((n.x, n.y), self.goal) and self.env.is_free(self.goal[0], self.goal[1]):
                best_dist = d
                best_idx = i

        if best_idx is not None:
            # append goal node
            goal_parent = best_idx
            goal_node = Node(self.goal[0], self.goal[1], parent=goal_parent,
                             cost=self.nodes[goal_parent].cost + self.distance((self.nodes[goal_parent].x, self.nodes[goal_parent].y), self.goal))
            goal_index = len(self.nodes)
            self.nodes.append(goal_node)
            self.all_edges.append(((self.nodes[goal_parent].x, self.nodes[goal_parent].y), (self.goal[0], self.goal[1])))
            print("Goal connected in final attempt")
            return self.reconstruct_path(goal_index)

        print("Goal NOT reached")
        return None

    # --------------------------
    # Path reconstruction
    # --------------------------

    def reconstruct_path(self, last_index):
        path = []
        idx = last_index
        while idx is not None:
            node = self.nodes[idx]
            path.append((node.x, node.y))
            idx = node.parent
        self.final_path = list(reversed(path))
        return self.final_path

    # --------------------------
    # Visualization
    # --------------------------

    def show_path(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.env.grid, cmap='binary', origin='lower',
                 extent=[0, self.env.width, 0, self.env.height])

        # Draw edges
        for (a, b) in self.all_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'c-', alpha=0.2)

        # Draw final path
        if self.final_path is not None:
            px = [p[0] for p in self.final_path]
            py = [p[1] for p in self.final_path]
            ax.plot(px, py, 'y-', linewidth=3, label="Final RRT* Path")
            ax.plot(px, py, 'yo')

        # Start/Goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=14, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=18, label='Goal')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title("RRT* Path Planning")
        ax.legend()
        plt.show()
