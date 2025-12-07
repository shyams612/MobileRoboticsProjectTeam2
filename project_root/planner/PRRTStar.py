# ============================================
# P-RRT* PLANNER (Potential Function Based RRT*)
# Based on: "Potential functions based sampling heuristic for optimal path planning"
# by Qureshi & Ayaz (2016)
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


class PRRTStar:
    def __init__(self,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 env: RandomEnvironment,
                 step_size: float = 1.5,
                 goal_radius: float = 0.71, # srqt(2) / 2
                 max_iters: int = 5000,
                 rewire_radius: float = 4.0,
                 early_stop: bool = True,

                 # P-RRT* specific parameters (Section 4, page 6)
                 k: int = 90,                    # Number of RGD iterations (controls exploitation)
                 lambda_step: float = 0.5,       # Small incremental step for RGD (λ in paper)
                 d_star_obs: float = 0.5):       # Distance threshold from obstacles (d*_obs in paper)

        self.start = start
        self.goal = goal
        self.env = env
        
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.goal_sample_rate = 0.05
        self.max_iters = max_iters
        self.rewire_radius = rewire_radius
        self.early_stop = early_stop
        
        # P-RRT* specific parameters
        self.k = k                          # RGD iterations (paper suggests 80-100)
        self.lambda_step = lambda_step      # Small step for gradient descent
        self.d_star_obs = d_star_obs        # Obstacle proximity threshold
        
        # Node list
        self.nodes: List[Node] = [
            Node(start[0], start[1], parent=None, cost=0.0)
        ]
        
        # Path will be stored after search
        self.final_path = None
        self.all_edges = []
        self.goal_node_idx = None

    # --------------------------
    # Utility
    # --------------------------

    def distance(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def sample_point(self):
        if random.random() < self.goal_sample_rate:
            return (self.goal[0], self.goal[1])
        return self.env.sample_free_point()

    def nearest_node_index(self, point):
        dists = [(node.x - point[0])**2 + (node.y - point[1])**2 for node in self.nodes]
        return int(np.argmin(dists))

    def steer(self, from_node: Node, to_point):
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
        neighbors = []
        for i, node in enumerate(self.nodes):
            if self.distance((node.x, node.y), (new_node.x, new_node.y)) <= self.rewire_radius:
                neighbors.append(i)
        return neighbors

    # --------------------------
    # P-RRT* Specific: Potential Field Functions
    # --------------------------
    
    def attractive_potential_gradient(self, point): 
        force_x = -2 * (point[0] - self.goal[0])
        force_y = -2 * (point[1] - self.goal[1])
        return (force_x, force_y)
    
    def nearest_obstacle_distance(self, point):
        x, y = point
        
        # Check if point itself is in obstacle
        if not self.env.is_free(x, y):
            return 0.0
        
        # Convert to grid coordinates
        grid_x = int(round(x))
        grid_y = int(round(y))
        
        # Only search up to slightly beyond d_star_obs
        # Key optimization: we only need to know if d_min ≤ d_star_obs
        max_search_cells = int(np.ceil(self.d_star_obs * 1.5))
        
        min_distance = float('inf')
        
        for dx in range(-max_search_cells, max_search_cells + 1):
            for dy in range(-max_search_cells, max_search_cells + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                # Check bounds
                if (check_x < 0 or check_x >= self.env.width or 
                    check_y < 0 or check_y >= self.env.height):
                    continue
                
                # Check if this cell is an obstacle
                if not self.env.is_free(check_x, check_y):
                    # Calculate actual Euclidean distance
                    dist = np.hypot(check_x - x, check_y - y)
                    min_distance = min(min_distance, dist)
                    
                    # Early termination: if we found an obstacle very close,
                    # we know d_min ≤ d_star_obs, which is all we need
                    if min_distance <= self.d_star_obs:
                        return min_distance
        
        # Return the minimum distance found (or large value if no obstacle nearby)
        if min_distance < float('inf'):
            return min_distance
        else:
            # No obstacle found within search radius
            return self.d_star_obs * 1.5
    
    def randomized_gradient_descent(self, x_rand):
        x_prand = x_rand
        
        # Algorithm 7, Line 2: Loop for k iterations
        for _ in range(self.k):
            F_att = self.attractive_potential_gradient(x_prand)
            d_min = self.nearest_obstacle_distance(x_prand)
            
            # Algorithm 7, Lines 5-6: Stop if too close to obstacle
            if d_min <= self.d_star_obs:
                return x_prand

            force_magnitude = np.hypot(F_att[0], F_att[1])
            if force_magnitude < self.goal_radius:
                return x_prand
            
            new_x = x_prand[0] + self.lambda_step * (F_att[0] / force_magnitude)
            new_y = x_prand[1] + self.lambda_step * (F_att[1] / force_magnitude)
            
            # Check if new position is valid
            if self.env.is_free(new_x, new_y):
                x_prand = (new_x, new_y)
            else:
                return x_prand
        
        return x_prand

    # --------------------------
    # Main search
    # --------------------------

    def search(self):
        """
        P-RRT* main search algorithm.
        
        From paper Algorithm 6 (page 5):
        Same as RRT* but with RGD(x_rand) to guide samples (Line 4).
        """
        for it in range(self.max_iters):
            if not it % 1000:
                print(f"Current Iteration: {it}")

            rand_point = self.sample_point()
            guided_point = self.randomized_gradient_descent(rand_point)

            # Find nearest node (to guided sample)
            nearest_idx = self.nearest_node_index(guided_point)
            nearest_node = self.nodes[nearest_idx]

            # Steer toward guided sample
            new_x, new_y = self.steer(nearest_node, guided_point)

            # Check collision
            if not self.env.is_free(new_x, new_y): continue
            if not self.env.is_straight_collision_free((nearest_node.x, nearest_node.y), (new_x, new_y)): continue

            new_node = Node(new_x, new_y, parent=nearest_idx,
                            cost=nearest_node.cost + self.distance((nearest_node.x, nearest_node.y),
                                                                   (new_x, new_y)))

            # Choose optimal parent among neighbors
            neighbors = self.get_neighbors(new_node)
            best_parent = nearest_idx
            best_cost = new_node.cost

            for i in neighbors:
                n = self.nodes[i]
                if self.env.is_straight_collision_free((n.x, n.y), (new_x, new_y)):
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

            # Rewire
            for i in neighbors:
                if i == best_parent:
                    continue

                n = self.nodes[i]
                new_cost = new_node.cost + self.distance((new_node.x, new_node.y),
                                                         (n.x, n.y))

                if new_cost < n.cost:
                    if self.env.is_straight_collision_free((new_node.x, new_node.y), (n.x, n.y)):
                        self.nodes[i].parent = new_index
                        self.nodes[i].cost = new_cost

            # Check if goal reached
            if self.distance((new_x, new_y), self.goal) < self.goal_radius:
                if not self.env.is_straight_collision_free((new_x, new_y), self.goal):
                    continue

                goal_cost = new_node.cost + self.distance((new_x, new_y), self.goal)
                
                if self.goal_node_idx is None:
                    goal_node = Node(self.goal[0], self.goal[1], parent=new_index, cost=goal_cost)
                    goal_index = len(self.nodes)
                    self.nodes.append(goal_node)
                    self.goal_node_idx = goal_index
                    self.all_edges.append(((new_x, new_y), (self.goal[0], self.goal[1])))
                    print(f"Goal reached at iteration {it} with cost {goal_cost:.2f}")
                    
                    if self.early_stop:
                        print("Early stopping enabled - terminating search")
                        return self.reconstruct_path(goal_index)
                
                elif goal_cost < self.nodes[self.goal_node_idx].cost:
                    self.nodes[self.goal_node_idx].parent = new_index
                    self.nodes[self.goal_node_idx].cost = goal_cost
                    print(f"Goal path improved at iteration {it} with cost {goal_cost:.2f}")

        # After all iterations
        print(f"Completed {self.max_iters} iterations")
        
        if self.goal_node_idx is not None:
            print(f"Final goal cost: {self.nodes[self.goal_node_idx].cost:.2f}")
            return self.reconstruct_path(self.goal_node_idx)
        
        # Final attempt: try to connect any existing node to the exact goal
        best_idx = None
        best_dist = float('inf')
        for i, n in enumerate(self.nodes):
            d = self.distance((n.x, n.y), self.goal)
            if d < best_dist and self.env.is_straight_collision_free((n.x, n.y), self.goal):
                best_dist = d
                best_idx = i

        if best_idx is not None:
            # append goal node
            goal_parent = best_idx
            goal_cost = self.nodes[goal_parent].cost + self.distance((self.nodes[goal_parent].x, self.nodes[goal_parent].y), self.goal)
            goal_node = Node(self.goal[0], self.goal[1], parent=goal_parent, cost=goal_cost)
            goal_index = len(self.nodes)
            self.nodes.append(goal_node)
            self.goal_node_idx = goal_index
            self.all_edges.append(((self.nodes[goal_parent].x, self.nodes[goal_parent].y), (self.goal[0], self.goal[1])))
            print(f"Goal connected in final attempt with cost {goal_cost:.2f}")
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
            ax.plot(px, py, 'y-', linewidth=3, label="Final P-RRT* Path")
            ax.plot(px, py, 'yo')

        # Start/Goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=14, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=18, label='Goal')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        
        title = f"P-RRT* Path Planning (k={self.k}, λ={self.lambda_step})"
        if self.early_stop:
            title += " [Early Stop: ON]"
        else:
            title += " [Early Stop: OFF]"
        ax.set_title(title)
        ax.legend()
        plt.show()