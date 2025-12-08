import numpy as np
import matplotlib.pyplot as plt
import dubins
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os

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
                 step_size: float = 3.0,
                 goal_radius: float = 3.0,
                 max_iters: int = 10000,
                 neighbor_radius: float = 10.0,
                 turning_radius: float = 2.0,
                 goal_sample_rate: float = 0.2,
                 early_stop: bool = False,
                 save_plots: bool = True,
                 save_dir: str = "plots/rrt_plots"):
        """
        RRT* for Ackermann car using Dubins paths
        
        Parameters:
        -----------
        start, goal: (x, y, theta)
        env: environment object (must implement is_free and is_collision_free)
        step_size: maximum distance along Dubins path per extension
        goal_radius: radius to consider goal reached
        max_iters: maximum number of iterations
        neighbor_radius: radius for rewiring
        turning_radius: minimum turning radius for Dubins curves
        goal_sample_rate: probability of sampling goal to bias tree
        early_stop: if True, stop when goal is first reached; if False, run for max_iters
        save_plots: if True, save plot every time goal is reached
        save_dir: directory to save plots
        """
        self.start = DubinsNode(*start, parent=None, cost=0.0)
        self.goal = DubinsNode(*goal, parent=None, cost=0.0)
        self.env = env
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iters = max_iters
        self.neighbor_radius = neighbor_radius
        self.turning_radius = turning_radius
        self.goal_sample_rate = goal_sample_rate
        self.early_stop = early_stop
        self.save_plots = save_plots
        self.save_dir = save_dir

        self.nodes: List[DubinsNode] = [self.start]
        self.all_edges = []
        self.final_path = None
        self.goal_node_idx = None  # Track if/when goal is added to tree
        self.save_counter = 0  # Counter for saved plots
        
        # Create save directory if saving is enabled
        if self.save_plots and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")

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
            if self.distance(node, new_node) <= self.neighbor_radius:
                neighbors.append(i)
        return neighbors

    # --------------------------
    # Main search
    # --------------------------

    def search(self):
        for it in range(self.max_iters):
            if it % 1000 == 0:
                print(f"Current Iteration: {it}")
            
            rand_node = self.sample_point()
            nearest_idx = self.nearest_node_index(rand_node)
            nearest_node = self.nodes[nearest_idx]

            new_node = self.steer(nearest_node, rand_node)

            if not self.env.is_dubins_collision_free(nearest_node, new_node, self.turning_radius):
                continue

            # Cost and parent selection
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

            euclidean_dist = np.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y)
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
        
        # Final attempt: try to connect any existing node to the exact goal
        best_idx = None
        best_dist = float('inf')
        for i, n in enumerate(self.nodes):
            d = self.distance(n, self.goal)
            if d < best_dist and self.env.is_dubins_collision_free(n, self.goal, self.turning_radius):
                best_dist = d
                best_idx = i

        if best_idx is not None:
            # append goal node
            goal_parent = best_idx
            goal_cost = self.nodes[goal_parent].cost + self.distance(self.nodes[goal_parent], self.goal)
            self.goal.parent = goal_parent
            self.goal.cost = goal_cost
            goal_index = len(self.nodes)
            self.nodes.append(self.goal)
            self.goal_node_idx = goal_index
            self.all_edges.append(((self.nodes[goal_parent].x, self.nodes[goal_parent].y, 
                                    self.nodes[goal_parent].theta),
                                   (self.goal.x, self.goal.y, self.goal.theta)))
            print(f"Goal connected in final attempt with cost {goal_cost:.2f}")
            
            # Save plot for final connection
            if self.save_plots:
                self.save_current_plot(self.max_iters, goal_cost, "final_connection")
            
            return self.reconstruct_path(goal_index)

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
        filename = f"rrt_iter{iteration:05d}_cost{cost:.2f}_{event_type}_{self.save_counter:03d}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
                            head_width=0.2, head_length=0.15,
                            fc='red', ec='red', alpha=0.4, linewidth=0.5)

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
        
        # Update title to reflect early_stop setting
        title = "RRT* with Dubins (Ackermann Vehicle)"
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