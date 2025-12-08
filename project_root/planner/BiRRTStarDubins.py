# ============================================
# BIDIRECTIONAL RRT* PLANNER WITH DUBINS PATHS
# Enhanced with auto-save and better visualization
# ============================================
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import dubins
import os

@dataclass
class DubinsNode:
    x: float
    y: float
    theta: float
    parent: Optional[int]
    cost: float = 0.0  # Cost from root to this node


class BidirectionalRRTStarDubins:
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
                 save_dir: str = "plots/birrt_plots",
                 save_dpi: int = 300):
        """
        Bidirectional RRT* for Ackermann car using Dubins paths
        
        Parameters:
        -----------
        start, goal: (x, y, theta)
        env: environment object (must implement is_free and is_collision_free)
        step_size: maximum distance along Dubins path per extension
        max_iters: maximum number of iterations
        goal_radius: distance threshold for connecting trees
        turning_radius: minimum turning radius for Dubins curves
        goal_sample_rate: probability of sampling goal to bias tree
        neighbor_radius: radius for finding neighbors and rewiring (RRT* parameter)
        early_stop: if True, stop when trees first connect; if False, continue optimizing
        save_plots: if True, save plot every time path is found/improved
        save_dir: directory to save plots
        save_dpi: resolution for saved plots (300 recommended for high quality)
        """
        self.start = start
        self.goal = goal
        self.env = env
        
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_radius = goal_radius
        self.turning_radius = turning_radius
        self.goal_sample_rate = goal_sample_rate
        self.neighbor_radius = neighbor_radius
        self.early_stop = early_stop
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.save_dpi = save_dpi
        
        # Two trees: one from start, one from goal
        self.start_tree: List[DubinsNode] = [
            DubinsNode(start[0], start[1], start[2], parent=None, cost=0.0)
        ]
        self.goal_tree: List[DubinsNode] = [
            DubinsNode(goal[0], goal[1], goal[2], parent=None, cost=0.0)
        ]
        
        # Path tracking
        self.final_path = None
        self.best_path = None
        self.best_cost = float('inf')
        self.start_edges = []
        self.goal_edges = []
        self.save_counter = 0
        
        # Create save directory if saving is enabled
        if self.save_plots and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")

    # --------------------------
    # Utility
    # --------------------------

    def dubins_distance(self, node1, node2):
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
        """Sample random configuration with heading"""
        if np.random.rand() < self.goal_sample_rate:
            return self.goal
        else:
            x, y = self.env.sample_free_point()
            theta = np.random.uniform(-np.pi, np.pi)
            return (x, y, theta)

    def nearest_node_index(self, tree: List[DubinsNode], point):
        """Find nearest node using Dubins distance"""
        point_node = DubinsNode(point[0], point[1], point[2], parent=None)
        dists = [self.dubins_distance(node, point_node) for node in tree]
        return int(np.argmin(dists))

    def near_nodes(self, tree: List[DubinsNode], point, radius):
        """Find all nodes within radius of point using Dubins distance"""
        point_node = DubinsNode(point[0], point[1], point[2], parent=None)
        near_indices = []
        for i, node in enumerate(tree):
            dist = self.dubins_distance(node, point_node)
            if dist <= radius:
                near_indices.append(i)
        return near_indices

    def steer(self, from_node: DubinsNode, to_point):
        """
        Return a new node along Dubins path from from_node toward to_point
        with maximum step_size distance.
        """
        q0 = (from_node.x, from_node.y, from_node.theta)
        
        if isinstance(to_point, tuple) and len(to_point) == 3:
            q1 = to_point
        else:
            dx = to_point[0] - from_node.x
            dy = to_point[1] - from_node.y
            estimated_theta = np.arctan2(dy, dx)
            q1 = (to_point[0], to_point[1], estimated_theta)
        
        path = dubins.shortest_path(q0, q1, self.turning_radius)

        if path.path_length() <= self.step_size:
            new_config = q1
        else:
            new_config = path.sample(self.step_size)

        return new_config

    def choose_parent(self, tree: List[DubinsNode], new_config, near_indices):
        """
        Choose the best parent from nearby nodes based on minimum cost.
        Returns (best_parent_idx, min_cost)
        """
        if not near_indices:
            return None, float('inf')
        
        best_parent = None
        min_cost = float('inf')
        new_node_temp = DubinsNode(new_config[0], new_config[1], new_config[2], parent=None)
        
        for idx in near_indices:
            node = tree[idx]
            edge_cost = self.dubins_distance(node, new_node_temp)
            total_cost = node.cost + edge_cost
            
            if total_cost < min_cost:
                if self.env.is_dubins_collision_free(node, new_node_temp, self.turning_radius):
                    best_parent = idx
                    min_cost = total_cost
        
        return best_parent, min_cost

    def rewire(self, tree: List[DubinsNode], new_idx: int, near_indices, edge_list):
        """
        Rewire the tree: check if routing through new node improves cost for nearby nodes.
        """
        new_node = tree[new_idx]
        
        for idx in near_indices:
            if idx == new_idx or idx == new_node.parent:
                continue
                
            node = tree[idx]
            edge_cost = self.dubins_distance(new_node, node)
            potential_cost = new_node.cost + edge_cost
            
            if potential_cost < node.cost:
                if self.env.is_dubins_collision_free(new_node, node, self.turning_radius):
                    if node.parent is not None:
                        old_parent = tree[node.parent]
                        old_edge = ((old_parent.x, old_parent.y, old_parent.theta), 
                                   (node.x, node.y, node.theta))
                        if old_edge in edge_list:
                            edge_list.remove(old_edge)
                    
                    node.parent = new_idx
                    node.cost = potential_cost
                    
                    edge_list.append(((new_node.x, new_node.y, new_node.theta), 
                                     (node.x, node.y, node.theta)))
                    
                    self._update_descendants_cost(tree, idx)

    def _update_descendants_cost(self, tree: List[DubinsNode], parent_idx: int):
        """Recursively update costs of all descendants after rewiring"""
        parent_node = tree[parent_idx]
        
        for i, node in enumerate(tree):
            if node.parent == parent_idx:
                edge_cost = self.dubins_distance(parent_node, node)
                node.cost = parent_node.cost + edge_cost
                self._update_descendants_cost(tree, i)

    def extend_tree_star(self, tree: List[DubinsNode], target_point, edge_list):
        """
        RRT* extension with Dubins paths: find best parent and rewire nearby nodes.
        Returns the index of the new node if successful, None otherwise.
        """
        nearest_idx = self.nearest_node_index(tree, target_point)
        nearest_node = tree[nearest_idx]

        new_config = self.steer(nearest_node, target_point)
        new_x, new_y, new_theta = new_config

        near_indices = self.near_nodes(tree, new_config, self.neighbor_radius)
        
        best_parent_idx, min_cost = self.choose_parent(tree, new_config, near_indices)
        
        if best_parent_idx is None:
            new_node_temp = DubinsNode(new_x, new_y, new_theta, parent=None)
            if self.env.is_dubins_collision_free(nearest_node, new_node_temp, self.turning_radius):
                best_parent_idx = nearest_idx
                edge_cost = self.dubins_distance(nearest_node, new_node_temp)
                min_cost = nearest_node.cost + edge_cost
            else:
                return None

        parent_node = tree[best_parent_idx]
        new_node = DubinsNode(new_x, new_y, new_theta, parent=best_parent_idx, cost=min_cost)
        new_index = len(tree)
        tree.append(new_node)
        
        edge_list.append(((parent_node.x, parent_node.y, parent_node.theta), 
                         (new_x, new_y, new_theta)))
        
        self.rewire(tree, new_index, near_indices, edge_list)
        
        return new_index

    def connect_trees(self, tree1: List[DubinsNode], tree2: List[DubinsNode], 
                     idx1: int, idx2: int):
        """
        Check if two nodes from different trees can be connected using Dubins path.
        """
        node1 = tree1[idx1]
        node2 = tree2[idx2]
        
        euclidean_dist = self.euclidean_distance(node1, node2)
        if euclidean_dist > self.goal_radius * 2:
            return False
        
        dubins_dist = self.dubins_distance(node1, node2)
        if dubins_dist <= self.goal_radius:
            if self.env.is_dubins_collision_free(node1, node2, self.turning_radius):
                return True
        return False

    # --------------------------
    # Main search
    # --------------------------

    def search(self):
        """
        Main bidirectional RRT* search algorithm with Dubins paths.
        """
        for it in range(self.max_iters):
            if it % 1000 == 0:
                print(f"Current Iteration: {it}")
            
            rand_point = self.sample_point()

            new_start_idx = self.extend_tree_star(self.start_tree, rand_point, 
                                                 self.start_edges)
            
            if new_start_idx is not None:
                new_start_node = self.start_tree[new_start_idx]
                new_start_config = (new_start_node.x, new_start_node.y, new_start_node.theta)
                
                new_goal_idx = self.extend_tree_star(self.goal_tree, new_start_config,
                                                    self.goal_edges)
                
                if new_goal_idx is not None:
                    if self.connect_trees(self.start_tree, self.goal_tree,
                                        new_start_idx, new_goal_idx):
                        connection_cost = self.dubins_distance(
                            self.start_tree[new_start_idx],
                            self.goal_tree[new_goal_idx]
                        )
                        total_cost = (self.start_tree[new_start_idx].cost + 
                                     self.goal_tree[new_goal_idx].cost + 
                                     connection_cost)
                        
                        if total_cost < self.best_cost:
                            self.best_cost = total_cost
                            self.best_path = self.reconstruct_path(new_start_idx, 
                                                                   new_goal_idx)
                            
                            if self.best_path is None:
                                print(f"Warning: Path reconstruction failed at iteration {it}")
                            else:
                                event_type = "first_connected" if self.final_path is None else "improved"
                                print(f"Trees connected at iteration {it} - Path cost: {total_cost:.2f}")
                                
                                if self.save_plots:
                                    self.final_path = self.best_path
                                    self.save_current_plot(it, total_cost, event_type)
                                
                                if self.early_stop:
                                    print("Early stopping enabled - terminating search")
                                    self.final_path = self.best_path
                                    return self.final_path

            # Swap trees
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
            self.start_edges, self.goal_edges = self.goal_edges, self.start_edges

        # Return best path found
        if self.best_path is not None:
            print(f"Search complete. Best path cost: {self.best_cost:.2f}")
            self.final_path = self.best_path
            
            if self.save_plots:
                self.save_current_plot(self.max_iters, self.best_cost, "final")
        else:
            print("Goal NOT reached")
        
        return self.final_path

    # --------------------------
    # Path reconstruction
    # --------------------------

    def reconstruct_path(self, start_idx: int, goal_idx: int):
        """
        Reconstruct path by tracing back through both trees.
        """
        # Build path from start tree
        path_from_start = []
        idx = start_idx
        while idx is not None:
            node = self.start_tree[idx]
            path_from_start.append((node.x, node.y, node.theta))
            idx = node.parent
        
        path_from_start = list(reversed(path_from_start))
        
        # Build path from goal tree
        path_from_goal = []
        idx = goal_idx
        while idx is not None:
            node = self.goal_tree[idx]
            path_from_goal.append((node.x, node.y, node.theta))
            idx = node.parent
        
        # Check which tree is actually the start tree
        start_dist = self.euclidean_distance(path_from_start[0], self.start)
        if start_dist < 0.01:
            path = path_from_start + path_from_goal
        else:
            path = list(reversed(path_from_goal)) + list(reversed(path_from_start))
        
        return path

    # --------------------------
    # Visualization and Saving
    # --------------------------

    def save_current_plot(self, iteration, cost, event_type):
        """Save the current plot state"""
        fig, ax = self._create_plot()
        
        filename = f"birrt_iter{iteration:05d}_cost{cost:.2f}_{event_type}_{self.save_counter:03d}.png"
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

        # Draw all nodes from start tree with dots and heading indicators
        for node in self.start_tree:
            ax.plot(node.x, node.y, 'b.', markersize=3, alpha=0.6)
            arrow_len = 0.8
            ax.arrow(node.x, node.y,
                    arrow_len*np.cos(node.theta), arrow_len*np.sin(node.theta),
                    head_width=0.2, head_length=0.15,
                    fc='blue', ec='blue', alpha=0.4, linewidth=0.5)

        # Draw all nodes from goal tree with dots and heading indicators
        for node in self.goal_tree:
            ax.plot(node.x, node.y, 'c.', markersize=3, alpha=0.6)
            arrow_len = 0.8
            ax.arrow(node.x, node.y,
                    arrow_len*np.cos(node.theta), arrow_len*np.sin(node.theta),
                    head_width=0.2, head_length=0.15,
                    fc='cyan', ec='cyan', alpha=0.4, linewidth=0.5)

        # Draw edges from start tree (blue)
        for (a, b) in self.start_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'b-', alpha=0.2, linewidth=0.5)

        # Draw edges from goal tree (cyan)
        for (a, b) in self.goal_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'c-', alpha=0.2, linewidth=0.5)

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
        ax.plot(self.start[0], self.start[1], 'go', markersize=14, label='Start')
        ax.arrow(self.start[0], self.start[1],
                2.0*np.cos(self.start[2]), 2.0*np.sin(self.start[2]),
                head_width=0.5, head_length=0.5, fc='green', ec='green')

        # Draw goal with heading arrow
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=18, label='Goal')
        ax.arrow(self.goal[0], self.goal[1],
                2.0*np.cos(self.goal[2]), 2.0*np.sin(self.goal[2]),
                head_width=0.5, head_length=0.5, fc='red', ec='red')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        
        title = "Bidirectional RRT* with Dubins (Ackermann Vehicle)"
        if self.early_stop:
            title += " [Early Stop: ON]"
        else:
            title += " [Early Stop: OFF]"
        
        if self.final_path:
            title += f"\nCost: {self.best_cost:.2f}"
        
        ax.set_title(title)
        ax.legend()
        
        return fig, ax

    def show_path(self):
        """Display the final plot"""
        fig, ax = self._create_plot()
        plt.show()