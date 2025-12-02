import numpy as np
import matplotlib.pyplot as plt
import dubins
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class DubinsNode:
    x: float
    y: float
    theta: float
    parent: Optional[int]
    cost: float  # cost from root to this node


class RRTStarDubins:
    def __init__(self,
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 env,
                 step_size: float = 2.0,
                 goal_radius: float = 2.0,
                 max_iters: int = 10000,
                 rewire_radius: float = 6.0,
                 turning_radius: float = 2.0,
                 goal_sample_rate: float = 0.1):
        """
        RRT* for Ackermann car using Dubins paths
        
        Parameters:
        -----------
        start, goal: (x, y, theta)
        env: environment object (must implement is_free and is_collision_free)
        step_size: maximum distance along Dubins path per extension
        goal_radius: radius to consider goal reached
        max_iters: max iterations
        rewire_radius: radius for rewiring
        turning_radius: minimum turning radius for Dubins curves
        goal_sample_rate: probability of sampling goal to bias tree
        """
        self.start = DubinsNode(*start, parent=None, cost=0.0)
        self.goal = DubinsNode(*goal, parent=None, cost=0.0)
        self.env = env
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iters = max_iters
        self.rewire_radius = rewire_radius
        self.turning_radius = turning_radius
        self.goal_sample_rate = goal_sample_rate

        self.nodes: List[DubinsNode] = [self.start]
        self.all_edges = []
        self.final_path = None

    # --------------------------
    # Utilities
    # --------------------------

    def distance(self, node1, node2):
        """Use Dubins path length as distance"""
        q0 = (node1.x, node1.y, node1.theta)
        q1 = (node2.x, node2.y, node2.theta)
        path = dubins.shortest_path(q0, q1, self.turning_radius)
        return path.path_length()

    def nearest_node_index(self, point):
        dists = [self.distance(node, point) for node in self.nodes]
        return int(np.argmin(dists))

    def sample_point(self):
        """Randomly sample (x,y,theta) in environment"""
        if np.random.rand() < self.goal_sample_rate:
            return self.goal
        else:
            x, y = self.env.sample_free_point()
            theta = np.random.uniform(-np.pi, np.pi)
            return DubinsNode(x, y, theta, parent=None, cost=0.0)

    def steer(self, from_node: DubinsNode, to_node: DubinsNode):
        """
        Return a new node along Dubins path from from_node toward to_node
        with maximum step_size distance.
        """
        q0 = (from_node.x, from_node.y, from_node.theta)
        q1 = (to_node.x, to_node.y, to_node.theta)
        path = dubins.shortest_path(q0, q1, self.turning_radius)

        # Sample along the Dubins path
        if path.path_length() <= self.step_size:
            new_config = q1
        else:
            # Sample configuration at distance step_size
            new_config = path.sample(self.step_size)

        return DubinsNode(x=new_config[0], y=new_config[1],
                        theta=new_config[2], parent=None, cost=0.0)

    def get_neighbors(self, new_node):
        neighbors = []
        for i, node in enumerate(self.nodes):
            if self.distance(node, new_node) <= self.rewire_radius:
                neighbors.append(i)
        return neighbors

    # --------------------------
    # Collision checking
    # --------------------------

    def is_collision_free(self, from_node, to_node):
        q0 = (from_node.x, from_node.y, from_node.theta)
        q1 = (to_node.x, to_node.y, to_node.theta)
        path = dubins.shortest_path(q0, q1, self.turning_radius)
        # Sample along Dubins path
        sample_distance = 0.5
        for i in np.arange(0, path.path_length(), sample_distance):
            x, y, theta = path.sample(i)
            if not self.env.is_free(x, y):
                return False
        return True

    # --------------------------
    # Main search
    # --------------------------

    def search(self):
        for it in range(self.max_iters):
            rand_node = self.sample_point()
            nearest_idx = self.nearest_node_index(rand_node)
            nearest_node = self.nodes[nearest_idx]

            new_node = self.steer(nearest_node, rand_node)

            if not self.is_collision_free(nearest_node, new_node):
                continue

            # Cost and parent selection
            neighbors = self.get_neighbors(new_node)
            best_parent = nearest_idx
            best_cost = nearest_node.cost + self.distance(nearest_node, new_node)

            for i in neighbors:
                n = self.nodes[i]
                if self.is_collision_free(n, new_node):
                    cost = n.cost + self.distance(n, new_node)
                    if cost < best_cost:
                        best_parent = i
                        best_cost = cost

            new_node.parent = best_parent
            new_node.cost = best_cost
            self.nodes.append(new_node)

            # Save edge for visualization
            self.all_edges.append(((self.nodes[best_parent].x, self.nodes[best_parent].y,
                                    self.nodes[best_parent].theta),
                                   (new_node.x, new_node.y, new_node.theta)))

            # Rewire neighbors
            for i in neighbors:
                n = self.nodes[i]
                new_cost = new_node.cost + self.distance(new_node, n)
                if new_cost < n.cost and self.is_collision_free(new_node, n):
                    n.parent = len(self.nodes) - 1
                    n.cost = new_cost

            # Check goal
            if self.distance(new_node, self.goal) < self.goal_radius:
                self.goal.parent = len(self.nodes) - 1
                self.goal.cost = new_node.cost + self.distance(new_node, self.goal)
                self.nodes.append(self.goal)
                print(f"Goal reached at iteration {it}")
                return self.reconstruct_path(len(self.nodes) - 1)
        print("Goal NOT reached")
        return None

    # --------------------------
    # Reconstruct path
    # --------------------------

    def reconstruct_path(self, last_index):
        path = []
        idx = last_index
        while idx is not None:
            node = self.nodes[idx]
            path.append((node.x, node.y, node.theta))
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

        # Draw all edges
        for (a, b) in self.all_edges:
            x0, y0, th0 = a
            x1, y1, th1 = b
            ax.plot([x0, x1], [y0, y1], 'c-', alpha=0.2)

        # Draw final path
        if self.final_path is not None:
            for i in range(len(self.final_path) - 1):
                q0 = self.final_path[i]
                q1 = self.final_path[i + 1]
                path = dubins.shortest_path(q0, q1, self.turning_radius)
                configurations, _ = path.sample_many(0.1)
                px = [c[0] for c in configurations]
                py = [c[1] for c in configurations]
                ax.plot(px, py, 'y-', linewidth=3)

                # Draw heading as small arrows along path
                for c in configurations[::5]:
                    ax.arrow(c[0], c[1],
                            0.5*np.cos(c[2]), 0.5*np.sin(c[2]),
                            head_width=0.3, head_length=0.3,
                            color='r')

        # Draw start with heading arrow
        ax.plot(self.start.x, self.start.y, 'go', markersize=14, label='Start')
        ax.arrow(self.start.x, self.start.y,
                2.0*np.cos(self.start.theta), 2.0*np.sin(self.start.theta),
                head_width=0.5, head_length=0.5, fc='green', ec='green')

        # Draw goal with optional heading arrow
        ax.plot(self.goal.x, self.goal.y, 'r*', markersize=18, label='Goal')
        ax.arrow(self.goal.x, self.goal.y,
                2.0*np.cos(self.goal.theta), 2.0*np.sin(self.goal.theta),
                head_width=0.5, head_length=0.5, fc='red', ec='red')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title("RRT* with Dubins (Ackermann Vehicle)")
        ax.legend()
        plt.show()

