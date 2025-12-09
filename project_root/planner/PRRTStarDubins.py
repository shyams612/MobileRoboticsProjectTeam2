# ============================================
# P-RRT* with Dubins (Ackermann Vehicle)
# ============================================

from dataclasses import dataclass
from typing import Optional, List, Tuple
from project_root.environment.RandomEnvironment import RandomEnvironment
import numpy as np
import matplotlib.pyplot as plt
import random
import dubins


@dataclass
class DubinsNode:
    x: float
    y: float
    theta: float
    parent: Optional[int]
    cost: float  # cost from root to this node


class PRRTStarDubins:
    def __init__(self,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 env: RandomEnvironment,
                 step_size: float = 2.0,
                 goal_radius: float = 2.0,
                 max_iters: int = 5000,
                 rewire_radius: float = 8.0,
                 turning_radius: float = 1.0,
                 goal_sample_rate: float = 0.05,
                 early_stop: bool = True,

                 # P-RRT* specific parameters
                 k: int = 90,             # RGD iterations (controls exploitation)
                 lambda_step: float = 0.5,
                 d_star_obs: float = 0.5):

        """
        P-RRT* with Dubins constraints:
        - start, goal: (x, y) (θ is internally chosen toward goal)
        - env: must implement sample_free_point(), is_free(), is_dubins_collision_free()
        """

        self.env = env

        # pure (x,y) for potential field & visualization
        self.start = start
        self.goal = goal

        # choose headings toward goal for start/goal states
        start_theta = np.arctan2(goal[1] - start[1], goal[0] - start[0])
        goal_theta = start_theta  # you can change this if you want a specific goal heading

        self.start_node = DubinsNode(start[0], start[1], start_theta, parent=None, cost=0.0)
        self.goal_node  = DubinsNode(goal[0],  goal[1],  goal_theta,  parent=None, cost=0.0)

        # RRT* parameters
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iters = max_iters
        self.rewire_radius = rewire_radius
        self.turning_radius = turning_radius
        self.goal_sample_rate = goal_sample_rate
        self.early_stop = early_stop

        # P-RRT* specific parameters
        self.k = k
        self.lambda_step = lambda_step
        self.d_star_obs = d_star_obs

        # Node list (DubinsNode)
        self.nodes: List[DubinsNode] = [self.start_node]

        # bookkeeping
        self.final_path = None
        self.all_edges = []      # list of ((x0,y0,theta0), (x1,y1,theta1))
        self.goal_node_idx = None

    # ------------------------------------------------------------
    # Utility: distances & sampling
    # ------------------------------------------------------------

    def dubins_distance(self, n1: DubinsNode, n2: DubinsNode) -> float:
        q0 = (n1.x, n1.y, n1.theta)
        q1 = (n2.x, n2.y, n2.theta)
        path = dubins.shortest_path(q0, q1, self.turning_radius)
        return path.path_length()

    def euclidean_xy(self, p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return np.hypot(p[0] - q[0], p[1] - q[1])

    def sample_point_xy(self) -> Tuple[float, float]:
        """Sample in R^2, goal-biased."""
        if random.random() < self.goal_sample_rate:
            return (self.goal[0], self.goal[1])
        return self.env.sample_free_point()

    def nearest_node_index(self, point_node: DubinsNode) -> int:
        dists = [self.dubins_distance(node, point_node) for node in self.nodes]
        return int(np.argmin(dists))

    def get_neighbors(self, new_node: DubinsNode) -> List[int]:
        neighbors = []
        for i, node in enumerate(self.nodes):
            if self.dubins_distance(node, new_node) <= self.rewire_radius:
                neighbors.append(i)
        return neighbors

    # ------------------------------------------------------------
    # P-RRT*: potential field utilities (2D, no heading)
    # ------------------------------------------------------------

    def attractive_potential_gradient(self, point: Tuple[float, float]) -> Tuple[float, float]:
        # pure attraction toward goal (no repulsion)
        force_x = -2 * (point[0] - self.goal[0])
        force_y = -2 * (point[1] - self.goal[1])
        return (force_x, force_y)

    def nearest_obstacle_distance(self, point: Tuple[float, float]) -> float:
        x, y = point

        # If inside obstacle already
        if not self.env.is_free(x, y):
            return 0.0

        grid_x = int(round(x))
        grid_y = int(round(y))

        max_search_cells = int(np.ceil(self.d_star_obs * 1.5))
        min_distance = float('inf')

        for dx in range(-max_search_cells, max_search_cells + 1):
            for dy in range(-max_search_cells, max_search_cells + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy

                if (check_x < 0 or check_x >= self.env.width or
                    check_y < 0 or check_y >= self.env.height):
                    continue

                if not self.env.is_free(check_x, check_y):
                    dist = np.hypot(check_x - x, check_y - y)
                    min_distance = min(min_distance, dist)
                    if min_distance <= self.d_star_obs:
                        return min_distance

        if min_distance < float('inf'):
            return min_distance
        else:
            return self.d_star_obs * 1.5

    def randomized_gradient_descent(self, x_rand: Tuple[float, float]) -> Tuple[float, float]:
        """
        RGD in R^2: iteratively moves the sample toward the goal along negative
        gradient of the attractive potential. Stops if near obstacle or near goal.
        """
        x_prand = x_rand

        for _ in range(self.k):
            F_att = self.attractive_potential_gradient(x_prand)
            d_min = self.nearest_obstacle_distance(x_prand)

            # if too close to obstacle, stop
            if d_min <= self.d_star_obs:
                return x_prand

            force_mag = np.hypot(F_att[0], F_att[1])
            if force_mag < self.goal_radius:
                return x_prand

            new_x = x_prand[0] + self.lambda_step * (F_att[0] / force_mag)
            new_y = x_prand[1] + self.lambda_step * (F_att[1] / force_mag)

            if self.env.is_free(new_x, new_y):
                x_prand = (new_x, new_y)
            else:
                return x_prand

        return x_prand

    def lift_to_dubins(self, xy: Tuple[float, float]) -> DubinsNode:
        """
        Convert a 2D point into a Dubins configuration by assigning a heading
        pointing approximately toward the goal.
        """
        x, y = xy
        theta = np.arctan2(self.goal[1] - y, self.goal[0] - x)
        return DubinsNode(x, y, theta, parent=None, cost=0.0)

    # ------------------------------------------------------------
    # Dubins steering
    # ------------------------------------------------------------

    def steer(self, from_node: DubinsNode, to_node: DubinsNode) -> DubinsNode:
        """
        Move from 'from_node' toward 'to_node' along a Dubins path
        by at most step_size.
        """
        q0 = (from_node.x, from_node.y, from_node.theta)
        q1 = (to_node.x, to_node.y, to_node.theta)

        path = dubins.shortest_path(q0, q1, self.turning_radius)

        if path.path_length() <= self.step_size:
            new_config = q1
        else:
            new_config = path.sample(self.step_size)

        return DubinsNode(
            x=new_config[0],
            y=new_config[1],
            theta=new_config[2],
            parent=None,
            cost=0.0
        )

    # ------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------

    def search(self):
        """
        P-RRT* + Dubins:
        - RGD guides samples in (x,y)
        - Tree grows with Dubins edges and RRT* rewiring
        """
        for it in range(self.max_iters):
            if it % 1000 == 0:
                print(f"Current Iteration: {it}")

            # 1) Sample in R^2
            rand_xy = self.sample_point_xy()

            # 2) Run RGD to bias sample toward goal
            guided_xy = self.randomized_gradient_descent(rand_xy)

            # 3) Lift to Dubins config (assign heading)
            guided_node = self.lift_to_dubins(guided_xy)

            # 4) Nearest in Dubins metric
            nearest_idx = self.nearest_node_index(guided_node)
            nearest_node = self.nodes[nearest_idx]

            # 5) Steer using Dubins path
            new_node = self.steer(nearest_node, guided_node)

            # 6) Collision check along Dubins curve
            if not self.env.is_dubins_collision_free(nearest_node, new_node, self.turning_radius):
                continue

            # 7) RRT* parent selection
            neighbors = self.get_neighbors(new_node)
            best_parent = nearest_idx
            best_cost = nearest_node.cost + self.dubins_distance(nearest_node, new_node)

            for i in neighbors:
                n = self.nodes[i]
                if self.env.is_dubins_collision_free(n, new_node, self.turning_radius):
                    temp_cost = n.cost + self.dubins_distance(n, new_node)
                    if temp_cost < best_cost:
                        best_parent = i
                        best_cost = temp_cost

            new_node.parent = best_parent
            new_node.cost = best_cost
            new_index = len(self.nodes)
            self.nodes.append(new_node)

            parent_node = self.nodes[best_parent]
            self.all_edges.append(((parent_node.x, parent_node.y, parent_node.theta),
                                   (new_node.x, new_node.y, new_node.theta)))

            # 8) Rewire neighbors
            for i in neighbors:
                if i == best_parent:
                    continue

                n = self.nodes[i]
                new_cost = new_node.cost + self.dubins_distance(new_node, n)
                if new_cost < n.cost:
                    if self.env.is_dubins_collision_free(new_node, n, self.turning_radius):
                        n.parent = new_index
                        n.cost = new_cost

            # 9) Goal check (Euclidean in x,y)
            if self.euclidean_xy((new_node.x, new_node.y), self.goal) < self.goal_radius:
                if not self.env.is_dubins_collision_free(new_node, self.goal_node, self.turning_radius):
                    continue

                goal_cost = new_node.cost + self.dubins_distance(new_node, self.goal_node)

                if self.goal_node_idx is None:
                    # First time reaching goal
                    self.goal_node.parent = new_index
                    self.goal_node.cost = goal_cost
                    self.nodes.append(self.goal_node)
                    self.goal_node_idx = len(self.nodes) - 1
                    self.all_edges.append(((new_node.x, new_node.y, new_node.theta),
                                           (self.goal_node.x, self.goal_node.y, self.goal_node.theta)))
                    print(f"Goal reached at iteration {it} with cost {goal_cost:.2f}")

                    if self.early_stop:
                        print("Early stopping enabled - terminating search")
                        return self.reconstruct_path(self.goal_node_idx)

                else:
                    # Possible improvement of existing goal path
                    if goal_cost < self.nodes[self.goal_node_idx].cost:
                        self.nodes[self.goal_node_idx].parent = new_index
                        self.nodes[self.goal_node_idx].cost = goal_cost
                        print(f"Goal path improved at iteration {it} with cost {goal_cost:.2f}")

        # After all iterations
        print(f"Completed {self.max_iters} iterations")
        if self.goal_node_idx is not None:
            print(f"Final goal cost: {self.nodes[self.goal_node_idx].cost:.2f}")
            return self.reconstruct_path(self.goal_node_idx)

        print("Goal NOT reached")
        return None

    # ------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------

    def reconstruct_path(self, last_index: int):
        path = []
        idx = last_index
        while idx is not None:
            n = self.nodes[idx]
            path.append((n.x, n.y, n.theta))
            idx = n.parent
        self.final_path = list(reversed(path))
        return self.final_path

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    def show_path(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.env.grid, cmap='binary', origin='lower',
                  extent=[0, self.env.width, 0, self.env.height])

        # draw all tree edges (light)
        for (a, b) in self.all_edges:
            x0, y0, th0 = a
            x1, y1, th1 = b
            ax.plot([x0, x1], [y0, y1], 'c-', alpha=0.2)

        # draw final Dubins path if available
        if self.final_path is not None:
            for i in range(len(self.final_path) - 1):
                q0 = self.final_path[i]
                q1 = self.final_path[i + 1]
                path = dubins.shortest_path(q0, q1, self.turning_radius)
                configs, _ = path.sample_many(0.1)
                px = [c[0] for c in configs]
                py = [c[1] for c in configs]
                ax.plot(px, py, 'y-', linewidth=3)

                # heading arrows along path
                for c in configs[::5]:
                    ax.arrow(c[0], c[1],
                             0.5 * np.cos(c[2]), 0.5 * np.sin(c[2]),
                             head_width=0.3, head_length=0.3,
                             color='r', alpha=0.7)

        # start
        ax.plot(self.start[0], self.start[1], 'go', markersize=14, label='Start')
        ax.arrow(self.start_node.x, self.start_node.y,
                 2.0 * np.cos(self.start_node.theta), 2.0 * np.sin(self.start_node.theta),
                 head_width=0.5, head_length=0.5, fc='green', ec='green')

        # goal
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=18, label='Goal')
        ax.arrow(self.goal_node.x, self.goal_node.y,
                 2.0 * np.cos(self.goal_node.theta), 2.0 * np.sin(self.goal_node.theta),
                 head_width=0.5, head_length=0.5, fc='red', ec='red')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')

        title = f"P-RRT* with Dubins (Ackermann Vehicle, R={self.turning_radius})"
        if self.early_stop:
            title += " [Early Stop: ON]"
        else:
            title += " [Early Stop: OFF]"
        ax.set_title(title)
        ax.legend()
        plt.show()