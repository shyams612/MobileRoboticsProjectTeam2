# ============================================
# BIDIRECTIONAL RRT* PLANNER WITH DUBINS PATHS
# ============================================
from dataclasses import dataclass
from typing import Optional, List, Tuple
from project_root.environment.RandomEnvironment import RandomEnvironment
import numpy as np
import matplotlib.pyplot as plt
import dubins

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
                 env: RandomEnvironment,
                 step_size: float = 2.0,
                 max_iters: int = 25000,
                 connection_threshold: float = 3.5,
                 turning_radius: float = 1.0,
                 goal_sample_rate: float = 0.1,
                 search_radius: float = 5.0):  # Radius for rewiring

        self.start = start
        self.goal = goal
        self.env = env
        
        self.step_size = step_size
        self.max_iters = max_iters
        self.connection_threshold = connection_threshold
        self.turning_radius = turning_radius
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        
        # Two trees: one from start, one from goal
        self.start_tree: List[DubinsNode] = [
            DubinsNode(start[0], start[1], start[2], parent=None, cost=0.0)
        ]
        self.goal_tree: List[DubinsNode] = [
            DubinsNode(goal[0], goal[1], goal[2], parent=None, cost=0.0)
        ]
        
        # Path will be stored after search
        self.final_path = None
        self.start_edges = []  # Edges from start tree
        self.goal_edges = []   # Edges from goal tree

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
            # Compute heading toward a reference point for more natural directions
            ref_x = (self.start[0] + self.goal[0]) / 2
            ref_y = (self.start[1] + self.goal[1]) / 2
            theta = np.arctan2(ref_y - y, ref_x - x) + np.random.uniform(-np.pi/4, np.pi/4)
            return (x, y, theta)

    def nearest_node_index(self, tree: List[DubinsNode], point):
        """Find nearest node using Dubins distance"""
        point_node = DubinsNode(point[0], point[1], point[2], parent=None)
        dists = [self.dubins_distance(node, point_node) for node in tree]
        return int(np.argmin(dists))

    def near_nodes(self, tree: List[DubinsNode], point, radius):
        """
        Find all nodes within radius of point using Dubins distance.
        """
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
        
        # If to_point is a tuple with 3 elements, use it directly
        # Otherwise, estimate a good heading based on direction of travel
        if isinstance(to_point, tuple) and len(to_point) == 3:
            q1 = to_point
        else:
            # Estimate heading: direction from current position to target
            dx = to_point[0] - from_node.x
            dy = to_point[1] - from_node.y
            estimated_theta = np.arctan2(dy, dx)
            q1 = (to_point[0], to_point[1], estimated_theta)
        
        path = dubins.shortest_path(q0, q1, self.turning_radius)

        # Sample along the Dubins path
        if path.path_length() <= self.step_size:
            new_config = q1
        else:
            # Sample configuration at distance step_size
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
            
            # Check if this path is better and collision-free
            if total_cost < min_cost:
                if self.env.is_dubins_collision_free(node, new_node_temp, self.turning_radius):
                    best_parent = idx
                    min_cost = total_cost
        
        return best_parent, min_cost

    def extend_tree_star(self, tree: List[DubinsNode], target_point, edge_list):
        """
        RRT* extension with Dubins paths: find best parent.
        Returns the index of the new node if successful, None otherwise.
        """
        # Find nearest node in tree
        nearest_idx = self.nearest_node_index(tree, target_point)
        nearest_node = tree[nearest_idx]

        # Steer toward target using Dubins
        new_config = self.steer(nearest_node, target_point)
        new_x, new_y, new_theta = new_config

        # Find near nodes
        near_indices = self.near_nodes(tree, new_config, self.search_radius)
        
        # Choose best parent from near nodes
        best_parent_idx, min_cost = self.choose_parent(tree, new_config, near_indices)
        
        if best_parent_idx is None:
            # Fallback to nearest if no valid parent found
            new_node_temp = DubinsNode(new_x, new_y, new_theta, parent=None)
            if self.env.is_dubins_collision_free(nearest_node, new_node_temp, self.turning_radius):
                best_parent_idx = nearest_idx
                edge_cost = self.dubins_distance(nearest_node, new_node_temp)
                min_cost = nearest_node.cost + edge_cost
            else:
                return None

        # Add new node with optimal parent
        parent_node = tree[best_parent_idx]
        new_node = DubinsNode(new_x, new_y, new_theta, parent=best_parent_idx, cost=min_cost)
        new_index = len(tree)
        tree.append(new_node)
        
        # Store edge for visualization
        edge_list.append(((parent_node.x, parent_node.y, parent_node.theta), 
                         (new_x, new_y, new_theta)))
        
        return new_index

    def connect_trees(self, tree1: List[DubinsNode], tree2: List[DubinsNode], 
                     idx1: int, idx2: int):
        """
        Check if two nodes from different trees can be connected using Dubins path.
        Returns True if connection is possible and collision-free.
        """
        node1 = tree1[idx1]
        node2 = tree2[idx2]
        
        # Check Euclidean distance first (faster)
        euclidean_dist = self.euclidean_distance(node1, node2)
        if euclidean_dist > self.connection_threshold * 2:
            return False
        
        # Check Dubins distance
        dubins_dist = self.dubins_distance(node1, node2)
        if dubins_dist <= self.connection_threshold:
            if self.env.is_dubins_collision_free(node1, node2, self.turning_radius):
                return True
        return False

    # --------------------------
    # Main search
    # --------------------------

    def search(self):
        for it in range(self.max_iters):
            if it % 1000 == 0:
                print(f"Current Iteration: {it}")
            # Sample random point
            rand_point = self.sample_point()

            # Extend start tree toward random point (with RRT* optimization)
            new_start_idx = self.extend_tree_star(self.start_tree, rand_point, 
                                                 self.start_edges)
            
            if new_start_idx is not None:
                # Try to connect goal tree to the new node in start tree
                new_start_node = self.start_tree[new_start_idx]
                new_start_config = (new_start_node.x, new_start_node.y, new_start_node.theta)
                
                # Extend goal tree toward the new start tree node
                new_goal_idx = self.extend_tree_star(self.goal_tree, new_start_config,
                                                    self.goal_edges)
                
                if new_goal_idx is not None:
                    # Check if trees can be connected
                    if self.connect_trees(self.start_tree, self.goal_tree,
                                        new_start_idx, new_goal_idx):
                        print(f"Trees connected at iteration {it}")
                        connection_cost = self.dubins_distance(
                            self.start_tree[new_start_idx],
                            self.goal_tree[new_goal_idx]
                        )
                        total_cost = (self.start_tree[new_start_idx].cost + 
                                     self.goal_tree[new_goal_idx].cost + 
                                     connection_cost)
                        print(f"Path cost: {total_cost:.2f}")
                        return self.reconstruct_path(new_start_idx, new_goal_idx)

            # Swap trees (alternate which tree extends first)
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
            self.start_edges, self.goal_edges = self.goal_edges, self.start_edges

        print("Goal NOT reached")
        return None

    # --------------------------
    # Path reconstruction
    # --------------------------

    def reconstruct_path(self, start_idx: int, goal_idx: int):
        """
        Reconstruct path by tracing back through both trees.
        Note: trees may have been swapped, so we need to check which is which.
        """
        # Build path from start tree
        path_from_start = []
        idx = start_idx
        while idx is not None:
            node = self.start_tree[idx]
            path_from_start.append((node.x, node.y, node.theta))
            idx = node.parent
        
        # Reverse to get start-to-connection order
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
            # start_tree is actually from start
            self.final_path = path_from_start + path_from_goal
        else:
            # Trees were swapped, so reverse
            self.final_path = list(reversed(path_from_goal)) + list(reversed(path_from_start))
        
        return self.final_path

    # --------------------------
    # Visualization
    # --------------------------

    def show_path(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.env.grid, cmap='binary', origin='lower',
                 extent=[0, self.env.width, 0, self.env.height])

        # Draw edges from start tree (blue)
        for (a, b) in self.start_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'b-', alpha=0.3, linewidth=0.5)

        # Draw edges from goal tree (green)
        for (a, b) in self.goal_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'g-', alpha=0.3, linewidth=0.5)

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
                            head_width=0.3, head_length=0.3,
                            color='r', alpha=0.7)

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
        ax.set_title("Bidirectional RRT* with Dubins Paths (Ackermann Vehicle)")
        ax.legend()
        plt.show()