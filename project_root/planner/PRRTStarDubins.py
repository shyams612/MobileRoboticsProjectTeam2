# ============================================
# P-RRT* PLANNER WITH DUBINS PATHS
# (Potential Function Based RRT* for Ackermann Vehicles)
# Based on: "Potential functions based sampling heuristic for optimal path planning"
# by Qureshi & Ayaz (2016)
# Enhanced with Dubins paths, auto-save, and better visualization
# ============================================
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import dubins
import random
import os

@dataclass
class DubinsNode:
    x: float
    y: float
    theta: float
    parent: Optional[int]
    cost: float = 0.0  # Cost from root to this node


class PRRTStarDubins:
    def __init__(self,
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 env,
                 step_size: float = 3.0,
                 goal_radius: float = 3.0,
                 max_iters: int = 3000,
                 rewire_radius: float = 10.0,
                 turning_radius: float = 2.0,
                 goal_sample_rate: float = 0.10,
                 early_stop: bool = False,
                 
                 # P-RRT* specific parameters (Section 4, page 6)
                 k: int = 90,                    # Number of RGD iterations (controls exploitation)
                 lambda_step: float = 0.5,       # Small incremental step for RGD (λ in paper)
                 d_star_obs: float = 1.0,        # Distance threshold from obstacles (d*_obs in paper)
                 
                 # Saving parameters
                 save_plots: bool = False,
                 save_dir: str = "prrt_plots",
                 save_dpi: int = 300):
        """
        P-RRT* for Ackermann car using Dubins paths with potential field guidance
        
        Parameters:
        -----------
        start, goal: (x, y, theta)
        env: environment object (must implement is_free and is_collision_free)
        step_size: maximum distance along Dubins path per extension
        goal_radius: radius to consider goal reached
        max_iters: maximum number of iterations
        rewire_radius: radius for rewiring (RRT* parameter)
        turning_radius: minimum turning radius for Dubins curves
        goal_sample_rate: probability of sampling goal directly
        early_stop: if True, stop when goal is first reached; if False, run for max_iters
        k: number of gradient descent iterations (paper suggests 80-100)
        lambda_step: step size for gradient descent
        d_star_obs: obstacle proximity threshold for RGD
        save_plots: if True, save plot every time goal is reached
        save_dir: directory to save plots
        save_dpi: resolution for saved plots
        """
        self.start = DubinsNode(*start, parent=None, cost=0.0)
        self.goal = DubinsNode(*goal, parent=None, cost=0.0)
        self.env = env
        
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.goal_sample_rate = goal_sample_rate
        self.max_iters = max_iters
        self.rewire_radius = rewire_radius
        self.turning_radius = turning_radius
        self.early_stop = early_stop
        
        # P-RRT* specific parameters
        self.k = k
        self.lambda_step = lambda_step
        self.d_star_obs = d_star_obs
        
        # Saving parameters
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.save_dpi = save_dpi
        
        # Node list
        self.nodes: List[DubinsNode] = [self.start]
        self.all_edges = []
        self.final_path = None
        self.goal_node_idx = None
        self.save_counter = 0
        
        # Create save directory if saving is enabled
        if self.save_plots and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")

    # --------------------------
    # Utilities
    # --------------------------

    def distance(self, node1, node2):
        """Use Dubins path length as distance"""
        q0 = (node1.x, node1.y, node1.theta) if isinstance(node1, DubinsNode) else node1
        q1 = (node2.x, node2.y, node2.theta) if isinstance(node2, DubinsNode) else node2
        path = dubins.shortest_path(q0, q1, self.turning_radius)
        return path.path_length()

    def euclidean_distance(self, a, b):
        """Euclidean distance (x,y only)"""
        ax, ay = (a.x, a.y) if isinstance(a, DubinsNode) else (a[0], a[1])
        bx, by = (b.x, b.y) if isinstance(b, DubinsNode) else (b[0], b[1])
        return np.hypot(ax - bx, ay - by)

    def sample_point(self):
        """Sample random configuration"""
        if random.random() < self.goal_sample_rate:
            return (self.goal.x, self.goal.y)
        return self.env.sample_free_point()

    def nearest_node_index(self, point):
        """Find nearest node using Dubins distance"""
        # Create temporary node with estimated heading
        dx = point[0] - self.nodes[0].x
        dy = point[1] - self.nodes[0].y
        est_theta = np.arctan2(dy, dx) if dx != 0 or dy != 0 else 0
        point_node = DubinsNode(point[0], point[1], est_theta, parent=None)
        
        dists = [self.distance(node, point_node) for node in self.nodes]
        return int(np.argmin(dists))

    def steer(self, from_node: DubinsNode, to_point):
        """
        Return a new node along Dubins path from from_node toward to_point
        with maximum step_size distance.
        """
        q0 = (from_node.x, from_node.y, from_node.theta)
        
        # Estimate heading for target point
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        estimated_theta = np.arctan2(dy, dx)
        q1 = (to_point[0], to_point[1], estimated_theta)
        
        path = dubins.shortest_path(q0, q1, self.turning_radius)

        if path.path_length() <= self.step_size:
            new_config = q1
        else:
            new_config = path.sample(self.step_size)

        return DubinsNode(x=new_config[0], y=new_config[1],
                        theta=new_config[2], parent=None, cost=0.0)

    def get_neighbors(self, new_node):
        """Find neighbors within rewire_radius"""
        neighbors = []
        for i, node in enumerate(self.nodes):
            if self.distance(node, new_node) <= self.rewire_radius:
                neighbors.append(i)
        return neighbors

    # --------------------------
    # P-RRT* Specific: Potential Field Functions
    # --------------------------
    
    def attractive_potential_gradient(self, point): 
        """Compute attractive force toward goal (x, y only)"""
        force_x = -2 * (point[0] - self.goal.x)
        force_y = -2 * (point[1] - self.goal.y)
        return (force_x, force_y)
    
    def nearest_obstacle_distance(self, point):
        """Find minimum distance to nearest obstacle"""
        x, y = point
        
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
    
    def randomized_gradient_descent(self, x_rand):
        """
        Apply gradient descent to guide sample toward goal while avoiding obstacles.
        Algorithm 7 from the paper.
        """
        x_prand = x_rand
        
        for _ in range(self.k):
            F_att = self.attractive_potential_gradient(x_prand)
            d_min = self.nearest_obstacle_distance(x_prand)
            
            # Stop if too close to obstacle
            if d_min <= self.d_star_obs:
                return x_prand

            force_magnitude = np.hypot(F_att[0], F_att[1])
            if force_magnitude < self.goal_radius:
                return x_prand
            
            # Take step in direction of attractive force
            new_x = x_prand[0] + self.lambda_step * (F_att[0] / force_magnitude)
            new_y = x_prand[1] + self.lambda_step * (F_att[1] / force_magnitude)
            
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
        P-RRT* main search algorithm with Dubins paths.
        Combines potential field guidance with RRT* optimization.
        """
        for it in range(self.max_iters):
            if it % 1000 == 0:
                print(f"Current Iteration: {it}")

            # Sample point and apply gradient descent
            rand_point = self.sample_point()
            guided_point = self.randomized_gradient_descent(rand_point)

            # Find nearest node
            nearest_idx = self.nearest_node_index(guided_point)
            nearest_node = self.nodes[nearest_idx]

            # Steer toward guided sample using Dubins
            new_node = self.steer(nearest_node, guided_point)

            # Check collision
            if not self.env.is_dubins_collision_free(nearest_node, new_node, self.turning_radius):
                continue

            # Cost and parent selection (RRT*)
            neighbors = self.get_neighbors(new_node)
            best_parent = nearest_idx
            best_cost = nearest_node.cost + self.distance(nearest_node, new_node)

            for i in neighbors:
                n = self.nodes[i]
                if self.env.is_dubins_collision_free(n, new_node, self.turning_radius):
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
                if new_cost < n.cost and self.env.is_dubins_collision_free(new_node, n, self.turning_radius):
                    n.parent = len(self.nodes) - 1
                    n.cost = new_cost

            # Check if goal reached
            euclidean_dist = self.euclidean_distance(new_node, self.goal)
            if euclidean_dist < self.goal_radius:
                if self.env.is_dubins_collision_free(new_node, self.goal, self.turning_radius):
                    goal_cost = new_node.cost + self.distance(new_node, self.goal)
                    
                    # If goal not yet in tree, add it
                    if self.goal_node_idx is None:
                        self.goal.parent = len(self.nodes) - 1
                        self.goal.cost = goal_cost
                        self.nodes.append(self.goal)
                        self.goal_node_idx = len(self.nodes) - 1
                        print(f"Goal first reached at iteration {it} with cost {goal_cost:.2f}")
                        
                        # Save plot when goal is first reached
                        if self.save_plots:
                            self.save_current_plot(it, goal_cost, "first_reached")
                        
                        # Early stop if enabled
                        if self.early_stop:
                            print("Early stopping enabled - terminating search")
                            return self.reconstruct_path(self.goal_node_idx)
                    
                    # If goal already in tree, check if this is a better connection
                    elif goal_cost < self.nodes[self.goal_node_idx].cost:
                        self.nodes[self.goal_node_idx].parent = len(self.nodes) - 1
                        self.nodes[self.goal_node_idx].cost = goal_cost
                        print(f"Goal path improved at iteration {it} with cost {goal_cost:.2f}")
                        
                        # Save plot when goal path is improved
                        if self.save_plots:
                            self.save_current_plot(it, goal_cost, "improved")

        # After all iterations complete
        print(f"Completed {self.max_iters} iterations")
        
        # If goal was reached during search, return that path
        if self.goal_node_idx is not None:
            print(f"Final goal cost: {self.nodes[self.goal_node_idx].cost:.2f}")
            
            # Save final plot
            if self.save_plots:
                self.save_current_plot(self.max_iters, self.nodes[self.goal_node_idx].cost, "final")
            
            return self.reconstruct_path(self.goal_node_idx)
        
        print("Goal NOT reached")
        return None
    
    # --------------------------
    # Path reconstruction
    # --------------------------

    def reconstruct_path(self, last_index):
        """Reconstruct path from goal to start"""
        path = []
        idx = last_index
        while idx is not None:
            node = self.nodes[idx]
            path.append((node.x, node.y, node.theta))
            idx = node.parent
        self.final_path = list(reversed(path))
        return self.final_path

    # --------------------------
    # Visualization and Saving
    # --------------------------

    def save_current_plot(self, iteration, cost, event_type):
        """Save the current plot state"""
        # Reconstruct current path
        if self.goal_node_idx is not None:
            self.reconstruct_path(self.goal_node_idx)
        
        # Generate plot
        fig, ax = self._create_plot()
        
        # Save with descriptive filename
        filename = f"prrt_iter{iteration:05d}_cost{cost:.2f}_{event_type}_{self.save_counter:03d}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=self.save_dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  → Plot saved: {filename}")
        self.save_counter += 1

    def _create_plot(self):
        """Create and return the plot figure"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.env.grid, cmap='binary', origin='lower',
                extent=[0, self.env.width, 0, self.env.height])

        # Draw all edges
        for (a, b) in self.all_edges:
            x0, y0, th0 = a
            x1, y1, th1 = b
            ax.plot([x0, x1], [y0, y1], 'c-', alpha=0.2)

        # Draw all sampled nodes with small dots and heading indicators
        for node in self.nodes:
            # Small dot for node position
            ax.plot(node.x, node.y, 'b.', markersize=3, alpha=0.6)
            # Small arrow for heading direction
            arrow_len = 0.8
            ax.arrow(node.x, node.y,
                    arrow_len*np.cos(node.theta), arrow_len*np.sin(node.theta),
                    head_width=0.2, head_length=0.15,
                    fc='blue', ec='blue', alpha=0.4, linewidth=0.5)

        # Draw final path with Dubins curves
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
                            head_width=0.2, head_length=0.15,
                            fc='red', ec='red', alpha=0.4, linewidth=0.5)

        # Draw start with heading arrow
        ax.plot(self.start.x, self.start.y, 'go', markersize=14, label='Start')
        ax.arrow(self.start.x, self.start.y,
                2.0*np.cos(self.start.theta), 2.0*np.sin(self.start.theta),
                head_width=0.5, head_length=0.5, fc='green', ec='green')

        # Draw goal with heading arrow
        ax.plot(self.goal.x, self.goal.y, 'r*', markersize=18, label='Goal')
        ax.arrow(self.goal.x, self.goal.y,
                2.0*np.cos(self.goal.theta), 2.0*np.sin(self.goal.theta),
                head_width=0.5, head_length=0.5, fc='red', ec='red')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        
        # Update title
        title = f"P-RRT* with Dubins (k={self.k}, λ={self.lambda_step})"
        if self.early_stop:
            title += " [Early Stop: ON]"
        else:
            title += " [Early Stop: OFF]"
        
        # Add cost info if path exists
        if self.final_path is not None and self.goal_node_idx is not None:
            title += f"\nCost: {self.nodes[self.goal_node_idx].cost:.2f}"
        
        ax.set_title(title)
        ax.legend()
        
        return fig, ax

    def show_path(self):
        """Display the final plot"""
        fig, ax = self._create_plot()
        plt.show()