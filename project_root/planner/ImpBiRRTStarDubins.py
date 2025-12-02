# ============================================
# BI-DIRECTIONAL RRT* WITH ARTIFICIAL POTENTIAL FIELD
# ============================================
from dataclasses import dataclass
from typing import Optional, List, Tuple
from project_root.environment.RandomEnvironment import RandomEnvironment
import numpy as np
import matplotlib.pyplot as plt
import random


@dataclass
class Node:
    x: float
    y: float
    parent: Optional[int]
    cost: float


class BiDirectionalRRTStarAPF:
    def __init__(self,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 env: RandomEnvironment,
                 step_size: float = 2.0,
                 goal_radius: float = 2.0,
                 max_iters: int = 2000,
                 rewire_radius: float = 4.0,
                 k_attractive: float = 0.5,
                 mu_repulsive: float = 10.0,
                 rho_influence: float = 15.0,
                 prob_connect: float = 0.1):
        """
        Improved Bi-Directional RRT* with Artificial Potential Field.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            env: Environment with obstacles
            step_size: Maximum step size for extending tree
            goal_radius: Radius around goal to consider as reached
            max_iters: Maximum iterations
            rewire_radius: Radius for rewiring
            k_attractive: Attractive force constant
            mu_repulsive: Repulsive force constant
            rho_influence: Obstacle radius of influence
            prob_connect: Probability of attempting tree connection
        """
        self.start = start
        self.goal = goal
        self.env = env
        
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.goal_sample_rate = 0.05
        self.max_iters = max_iters
        self.rewire_radius = rewire_radius
        
        # APF parameters
        self.k_attractive = k_attractive
        self.mu_repulsive = mu_repulsive
        self.rho_influence = rho_influence
        self.prob_connect = prob_connect
        
        # Two trees: 0 grows from start, 1 grows from goal
        self.trees: List[List[Node]] = [
            [Node(start[0], start[1], parent=None, cost=0.0)],
            [Node(goal[0], goal[1], parent=None, cost=0.0)]
        ]
        
        self.final_path = None
        self.all_edges = [[], []]  # Edges for each tree
        self.best_cost = float('inf')
        self.connection_point = None
        
    # --------------------------
    # Utility Methods
    # --------------------------
    
    def distance(self, a, b):
        """Euclidean distance between two points."""
        return np.hypot(a[0] - b[0], a[1] - b[1])
    
    def sample_point(self):
        """Sample a random point, with bias toward goal."""
        if random.random() < self.goal_sample_rate:
            if self.env.is_free(self.goal[0], self.goal[1]):
                return (self.goal[0], self.goal[1])
        return self.env.sample_free_point()
    
    def nearest_node_index(self, tree_idx: int, point: Tuple[float, float]) -> int:
        """Find nearest node in specified tree to the given point."""
        tree = self.trees[tree_idx]
        dists = [(node.x - point[0])**2 + (node.y - point[1])**2 
                 for node in tree]
        return int(np.argmin(dists))
    
    # --------------------------
    # Artificial Potential Field Methods
    # --------------------------
    
    def calculate_repulsive_force(self, point: Tuple[float, float]) -> np.ndarray:
        """Calculate repulsive force from obstacles."""
        repulsive_force = np.zeros(2)
        x = np.array(point)
        
        # Get obstacles from environment
        obstacles = self.get_obstacles_from_env()
        
        for obs in obstacles:
            # Find closest point on obstacle to current point
            min_x = max(obs[0], min(x[0], obs[2]))
            min_y = max(obs[1], min(x[1], obs[3]))
            closest_point = np.array([min_x, min_y])
            distance = np.linalg.norm(x - closest_point)
            
            if distance <= self.rho_influence and distance > 0:
                direction = x - closest_point
                repulsive_force += (self.mu_repulsive / 2 * 
                                   ((1 / distance) - (1 / self.rho_influence)) ** 2 * 
                                   (direction / distance ** 3))
        
        return repulsive_force
    
    def calculate_potential_force(self, 
                                  start: Tuple[float, float], 
                                  end: Tuple[float, float]) -> np.ndarray:
        """Calculate total potential force (attractive + repulsive)."""
        direction = np.array(end) - np.array(start)
        distance = np.linalg.norm(direction)
        
        if distance == 0:
            return np.zeros(2)
        
        # Attractive force toward target
        attractive_force = self.k_attractive * direction / distance
        
        # Repulsive force from obstacles
        repulsive_force = self.calculate_repulsive_force(start)
        
        total_force = attractive_force + repulsive_force
        return total_force
    
    def get_obstacles_from_env(self) -> List[Tuple[float, float, float, float]]:
        """Extract obstacle bounding boxes from environment."""
        obstacles = []
        grid = self.env.grid
        
        # Find contiguous obstacle regions (simplified approach)
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 1 and not visited[i, j]:
                    # Found an obstacle cell, create bounding box
                    min_i, max_i = i, i
                    min_j, max_j = j, j
                    
                    # Simple flood to find extent (limited to avoid performance issues)
                    for di in range(10):
                        for dj in range(10):
                            ni, nj = i + di, j + dj
                            if (ni < grid.shape[0] and nj < grid.shape[1] and 
                                grid[ni, nj] == 1):
                                max_i = max(max_i, ni)
                                max_j = max(max_j, nj)
                                visited[ni, nj] = True
                    
                    obstacles.append((min_j, min_i, max_j + 1, max_i + 1))
        
        return obstacles
    
    def steer(self, from_node: Node, to_point: Tuple[float, float]) -> Tuple[float, float]:
        """Steer from node toward point with APF influence."""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = np.hypot(dx, dy)
        
        if dist == 0:
            return (from_node.x, from_node.y)
        
        # Calculate direction
        if dist <= self.step_size:
            new_x, new_y = to_point
        else:
            u = np.array([dx / dist, dy / dist])
            base_point = np.array([from_node.x, from_node.y]) + u * self.step_size
            new_x, new_y = base_point[0], base_point[1]
        
        # Apply potential field force
        potential_force = self.calculate_potential_force(
            (from_node.x, from_node.y), 
            (new_x, new_y)
        )
        
        # Adjust position with potential force (scaled)
        force_scale = 0.3
        new_x += potential_force[0] * force_scale
        new_y += potential_force[1] * force_scale
        
        return (new_x, new_y)
    
    def get_neighbors(self, tree_idx: int, new_node: Node) -> List[int]:
        """Get all nodes within rewire radius in specified tree."""
        tree = self.trees[tree_idx]
        neighbors = []
        for i, node in enumerate(tree):
            if self.distance((node.x, node.y), (new_node.x, new_node.y)) <= self.rewire_radius:
                neighbors.append(i)
        return neighbors
    
    # --------------------------
    # Main Search Algorithm
    # --------------------------
    
    def search(self):
        """Run bi-directional RRT* with APF."""
        active_tree = 0  # Start with tree growing from start
        
        for it in range(self.max_iters):
            # Sample random point
            rand_point = self.sample_point()
            
            # Extend active tree
            self.extend_tree(active_tree, rand_point)
            
            # Try to connect trees with probability
            if random.random() < self.prob_connect:
                if self.try_connect_trees():
                    print(f"Trees connected at iteration {it}")
                    if self.final_path is not None:
                        return self.final_path
            
            # Swap active tree
            active_tree = 1 - active_tree
            
            # Check if we've found a good enough path
            if self.best_cost < float('inf') and random.random() < 0.01:
                print(f"Path found with cost {self.best_cost:.2f} at iteration {it}")
                return self.final_path
        
        print(f"Max iterations reached. Best cost: {self.best_cost:.2f}")
        return self.final_path
    
    def extend_tree(self, tree_idx: int, rand_point: Tuple[float, float]):
        """Extend the specified tree toward the random point."""
        # Find nearest node
        nearest_idx = self.nearest_node_index(tree_idx, rand_point)
        nearest_node = self.trees[tree_idx][nearest_idx]
        
        # Steer with APF
        new_x, new_y = self.steer(nearest_node, rand_point)
        
        # Check collision
        if not self.env.is_free(new_x, new_y):
            return
        if not self.env.is_straight_collision_free(
            (nearest_node.x, nearest_node.y), (new_x, new_y)):
            return
        
        # Create new node
        new_node = Node(
            new_x, new_y, 
            parent=nearest_idx,
            cost=nearest_node.cost + self.distance(
                (nearest_node.x, nearest_node.y), (new_x, new_y)
            )
        )
        
        # Choose optimal parent
        neighbors = self.get_neighbors(tree_idx, new_node)
        best_parent = nearest_idx
        best_cost = new_node.cost
        
        for i in neighbors:
            n = self.trees[tree_idx][i]
            if self.env.is_straight_collision_free((n.x, n.y), (new_x, new_y)):
                temp_cost = n.cost + self.distance((n.x, n.y), (new_x, new_y))
                if temp_cost < best_cost:
                    best_parent = i
                    best_cost = temp_cost
        
        new_node.parent = best_parent
        new_node.cost = best_cost
        
        # Add node to tree
        new_index = len(self.trees[tree_idx])
        self.trees[tree_idx].append(new_node)
        
        # Store edge
        parent_node = self.trees[tree_idx][best_parent]
        self.all_edges[tree_idx].append(
            ((parent_node.x, parent_node.y), (new_x, new_y))
        )
        
        # Rewire
        for i in neighbors:
            if i == best_parent:
                continue
            
            n = self.trees[tree_idx][i]
            new_cost = new_node.cost + self.distance(
                (new_node.x, new_node.y), (n.x, n.y)
            )
            
            if new_cost < n.cost:
                if self.env.is_straight_collision_free(
                    (new_node.x, new_node.y), (n.x, n.y)):
                    self.trees[tree_idx][i].parent = new_index
                    self.trees[tree_idx][i].cost = new_cost
    
    def try_connect_trees(self) -> bool:
        """Attempt to connect the two trees."""
        # Try to connect nearest nodes from both trees
        min_dist = float('inf')
        best_pair = None
        
        for i, node1 in enumerate(self.trees[0]):
            for j, node2 in enumerate(self.trees[1]):
                dist = self.distance((node1.x, node1.y), (node2.x, node2.y))
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j)
        
        if best_pair is None:
            return False
        
        idx0, idx1 = best_pair
        node0 = self.trees[0][idx0]
        node1 = self.trees[1][idx1]
        
        # Check if connection is collision-free
        if not self.env.is_straight_collision_free(
            (node0.x, node0.y), (node1.x, node1.y)):
            return False
        
        # Calculate total cost
        total_cost = node0.cost + node1.cost + self.distance(
            (node0.x, node0.y), (node1.x, node1.y)
        )
        
        if total_cost < self.best_cost:
            self.best_cost = total_cost
            self.connection_point = ((node0.x, node0.y), (node1.x, node1.y))
            
            # Reconstruct path
            path1 = self.reconstruct_path_in_tree(0, idx0)
            path2 = self.reconstruct_path_in_tree(1, idx1)
            path2.reverse()
            
            self.final_path = path1 + path2
            return True
        
        return False
    
    def reconstruct_path_in_tree(self, tree_idx: int, node_idx: int) -> List[Tuple[float, float]]:
        """Reconstruct path from root to node in specified tree."""
        path = []
        idx = node_idx
        
        while idx is not None:
            node = self.trees[tree_idx][idx]
            path.append((node.x, node.y))
            idx = node.parent
        
        return list(reversed(path))
    
    def reconstruct_path(self, last_index: int):
        """Legacy method for compatibility."""
        return self.final_path
    
    # --------------------------
    # Visualization
    # --------------------------
    
    def show_path(self):
        """Visualize the bi-directional RRT* trees and final path."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.env.grid, cmap='binary', origin='lower',
                 extent=[0, self.env.width, 0, self.env.height])
        
        # Draw tree 0 (from start) in cyan
        for (a, b) in self.all_edges[0]:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'c-', alpha=0.3, linewidth=0.5)
        
        # Draw tree 1 (from goal) in magenta
        for (a, b) in self.all_edges[1]:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'm-', alpha=0.3, linewidth=0.5)
        
        # Draw connection if exists
        if self.connection_point:
            ax.plot([self.connection_point[0][0], self.connection_point[1][0]],
                   [self.connection_point[0][1], self.connection_point[1][1]],
                   'g--', linewidth=2, alpha=0.7, label='Tree Connection')
        
        # Draw final path
        if self.final_path is not None:
            px = [p[0] for p in self.final_path]
            py = [p[1] for p in self.final_path]
            ax.plot(px, py, 'y-', linewidth=3, label=f"Final Path (cost={self.best_cost:.2f})")
            ax.plot(px, py, 'yo', markersize=4)
        
        # Start/Goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=14, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=18, label='Goal')
        
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title("Bi-Directional RRT* with Artificial Potential Field")
        ax.legend()
        plt.show()