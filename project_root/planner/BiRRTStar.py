# ============================================
# BIDIRECTIONAL RRT* PLANNER
# ============================================
from dataclasses import dataclass
from typing import Optional, List, Tuple
from project_root.environment.RandomEnvironment import RandomEnvironment
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Node:
    x: float
    y: float
    parent: Optional[int]
    cost: float = 0.0  # Cost from root to this node


class BidirectionalRRTStar:
    def __init__(self,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 env: RandomEnvironment,
                 step_size: float = 1.0,
                 max_iters: int = 25000,
                 connection_threshold: float = 1.5,
                 search_radius: float = 3.0): 

        self.start = start
        self.goal = goal
        self.env = env
        
        self.step_size = step_size
        self.max_iters = max_iters
        self.connection_threshold = connection_threshold
        self.search_radius = search_radius
        
        # Two trees: one from start, one from goal
        self.start_tree: List[Node] = [
            Node(start[0], start[1], parent=None, cost=0.0)
        ]
        self.goal_tree: List[Node] = [
            Node(goal[0], goal[1], parent=None, cost=0.0)
        ]
        
        # Path will be stored after search
        self.final_path = None
        self.start_edges = []  # Edges from start tree
        self.goal_edges = []   # Edges from goal tree

    # --------------------------
    # Utility
    # --------------------------

    def distance(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def sample_point(self):
        return self.env.sample_free_point()

    def nearest_node_index(self, tree: List[Node], point):
        dists = [(node.x - point[0])**2 + (node.y - point[1])**2
                 for node in tree]
        return int(np.argmin(dists))

    def near_nodes(self, tree: List[Node], point, radius):
        """
        Find all nodes within radius of point.
        """
        near_indices = []
        for i, node in enumerate(tree):
            dist = self.distance((node.x, node.y), point)
            if dist <= radius:
                near_indices.append(i)
        return near_indices

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

    def choose_parent(self, tree: List[Node], new_pos, near_indices):
        """
        Choose the best parent from nearby nodes based on minimum cost.
        Returns (best_parent_idx, min_cost)
        """
        if not near_indices:
            return None, float('inf')
        
        best_parent = None
        min_cost = float('inf')
        
        for idx in near_indices:
            node = tree[idx]
            edge_cost = self.distance((node.x, node.y), new_pos)
            total_cost = node.cost + edge_cost
            
            # Check if this path is better and collision-free
            if total_cost < min_cost:
                if self.env.is_straight_collision_free((node.x, node.y), new_pos):
                    best_parent = idx
                    min_cost = total_cost
        
        return best_parent, min_cost

    def extend_tree_star(self, tree: List[Node], target_point, edge_list):
        """
        RRT* extension: find best parent.
        Returns the index of the new node if successful, None otherwise.
        """
        # Find nearest node in tree
        nearest_idx = self.nearest_node_index(tree, target_point)
        nearest_node = tree[nearest_idx]

        # Steer toward target
        new_x, new_y = self.steer(nearest_node, target_point)

        # Check collision for new point
        if not self.env.is_free(new_x, new_y):
            return None

        # Find near nodes
        near_indices = self.near_nodes(tree, (new_x, new_y), self.search_radius)
        
        # Choose best parent from near nodes
        best_parent_idx, min_cost = self.choose_parent(tree, (new_x, new_y), 
                                                       near_indices)
        
        if best_parent_idx is None:
            # Fallback to nearest if no valid parent found
            if self.env.is_straight_collision_free((nearest_node.x, nearest_node.y),
                                                   (new_x, new_y)):
                best_parent_idx = nearest_idx
                edge_cost = self.distance((nearest_node.x, nearest_node.y), 
                                        (new_x, new_y))
                min_cost = nearest_node.cost + edge_cost
            else:
                return None

        # Add new node with optimal parent
        parent_node = tree[best_parent_idx]
        new_node = Node(new_x, new_y, parent=best_parent_idx, cost=min_cost)
        new_index = len(tree)
        tree.append(new_node)
        
        # Store edge for visualization
        edge_list.append(((parent_node.x, parent_node.y), (new_x, new_y)))
        
        return new_index

    def connect_trees(self, tree1: List[Node], tree2: List[Node], 
                     idx1: int, idx2: int):
        """
        Check if two nodes from different trees can be connected.
        Returns True if connection is possible and collision-free.
        """
        node1 = tree1[idx1]
        node2 = tree2[idx2]
        
        dist = self.distance((node1.x, node1.y), (node2.x, node2.y))
        
        if dist <= self.connection_threshold:
            if self.env.is_straight_collision_free((node1.x, node1.y), 
                                         (node2.x, node2.y)):
                return True
        return False

    # --------------------------
    # Main search
    # --------------------------

    def search(self):
        for it in range(self.max_iters):
            # Sample random point
            rand_point = self.sample_point()

            # Extend start tree toward random point (with RRT* optimization)
            new_start_idx = self.extend_tree_star(self.start_tree, rand_point, 
                                                 self.start_edges)
            
            if new_start_idx is not None:
                # Try to connect goal tree to the new node in start tree
                new_start_node = self.start_tree[new_start_idx]
                new_start_pos = (new_start_node.x, new_start_node.y)
                
                # Extend goal tree toward the new start tree node
                new_goal_idx = self.extend_tree_star(self.goal_tree, new_start_pos,
                                                    self.goal_edges)
                
                if new_goal_idx is not None:
                    # Check if trees can be connected
                    if self.connect_trees(self.start_tree, self.goal_tree,
                                        new_start_idx, new_goal_idx):
                        print(f"Trees connected at iteration {it}")
                        connection_cost = self.distance(
                            (self.start_tree[new_start_idx].x, 
                             self.start_tree[new_start_idx].y),
                            (self.goal_tree[new_goal_idx].x,
                             self.goal_tree[new_goal_idx].y)
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
        """
        # Build path from start tree
        path_from_start = []
        idx = start_idx
        while idx is not None:
            node = self.start_tree[idx]
            path_from_start.append((node.x, node.y))
            idx = node.parent
        
        # Reverse to get start-to-connection order
        path_from_start = list(reversed(path_from_start))
        
        # Build path from goal tree
        path_from_goal = []
        idx = goal_idx
        while idx is not None:
            node = self.goal_tree[idx]
            path_from_goal.append((node.x, node.y))
            idx = node.parent
        
        # Check which tree is actually the start tree
        if self.distance(path_from_start[0], self.start) < 0.01:
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

        # Draw final path
        if self.final_path is not None:
            px = [p[0] for p in self.final_path]
            py = [p[1] for p in self.final_path]
            ax.plot(px, py, 'y-', linewidth=3, label="Final Bi-RRT* Path")
            ax.plot(px, py, 'yo', markersize=4)

        # Start/Goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=14, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=18, label='Goal')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title("Bidirectional RRT* Path Planning")
        ax.legend()
        plt.show()