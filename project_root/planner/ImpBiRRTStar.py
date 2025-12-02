# ============================================
# IMPROVED BIDIRECTIONAL RRT* WITH APF
# Based on "Improved Bidirectional RRT* Algorithm for Robot Path Planning"
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
    cost: float = 0.0


class ImprovedBidirectionalRRTStar:
    def __init__(self,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 env: RandomEnvironment,
                 step_size: float = 1.5,
                 max_iters: int = 25000,
                 connection_threshold: float = 1.5,
                 search_radius: float = 3.0,
                 apf_attraction_coeff: float = 0.5,
                 apf_repulsion_coeff: float = 2.0,
                 apf_obstacle_influence: float = 3.0):

        self.start = start
        self.goal = goal
        self.env = env
        
        self.step_size = step_size
        self.max_iters = max_iters
        self.connection_threshold = connection_threshold
        self.search_radius = search_radius
        
        # APF parameters
        self.apf_xi = apf_attraction_coeff  # Gravitational coefficient
        self.apf_mu = apf_repulsion_coeff   # Repulsion coefficient
        self.apf_rho0 = apf_obstacle_influence  # Obstacle influence range
        
        # Two trees: one from start, one from goal
        self.start_tree: List[Node] = [
            Node(start[0], start[1], parent=None, cost=0.0)
        ]
        self.goal_tree: List[Node] = [
            Node(goal[0], goal[1], parent=None, cost=0.0)
        ]
        
        self.final_path = None
        self.start_edges = []
        self.goal_edges = []
        
        # Track which tree started as start/goal for proper swapping
        self.start_tree_is_from_start = True

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
        """Find all nodes within radius of point."""
        near_indices = []
        for i, node in enumerate(tree):
            dist = self.distance((node.x, node.y), point)
            if dist <= radius:
                near_indices.append(i)
        return near_indices

    def steer(self, from_node: Node, to_point):
        """Return a new node that is STEP_SIZE distance toward target point."""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = np.hypot(dx, dy)

        if dist <= self.step_size:
            new_x, new_y = to_point
        else:
            new_x = from_node.x + (dx / dist) * self.step_size
            new_y = from_node.y + (dy / dist) * self.step_size

        return new_x, new_y

    # --------------------------
    # Artificial Potential Field Methods (Section 3.1)
    # Based on paper equations (1), (2), (3)
    # --------------------------

    def get_nearby_obstacles_5x5(self, point):
        """
        Get obstacle positions in a 5x5 grid around the point (Figure 3).
        Returns list of (x, y) positions of obstacles.
        """
        obstacles = []
        px, py = int(point[0]), int(point[1])
        
        # 5x5 search range as shown in Figure 3
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_x = px + dx
                check_y = py + dy
                
                if (0 <= check_x < self.env.width and 
                    0 <= check_y < self.env.height):
                    if not self.env.is_free(check_x, check_y):
                        obstacles.append((check_x, check_y))
        
        return obstacles

    def compute_attractive_potential_gradient(self, current_pos, goal_pos):
        """
        Compute gradient of attractive potential (Equation 1).
        U_att = (1/2) * ξ * ρ²(P, P_goal)
        ∇U_att = ξ * ρ(P, P_goal) * ∇ρ(P, P_goal)
        
        This gives us F_att = -∇U_att pointing toward goal.
        """
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        dist = np.hypot(dx, dy)
        
        if dist < 0.01:
            return (0.0, 0.0)
        
        # Gradient of attractive potential (negative gradient = attractive force)
        # F_att = -ξ * dist * (direction to goal)
        f_att_x = self.apf_xi * dx  # proportional to distance
        f_att_y = self.apf_xi * dy
        
        return (f_att_x, f_att_y)

    def compute_repulsive_potential_gradient(self, current_pos):
        """
        Compute gradient of repulsive potential (Equation 2).
        U_rep = (1/2) * μ * (1/ρ - 1/ρ₀)² if ρ ≤ ρ₀, else 0
        
        Gradient gives us F_rep pushing away from obstacles.
        """
        obstacles = self.get_nearby_obstacles_5x5(current_pos)
        f_rep_x = 0.0
        f_rep_y = 0.0
        
        for obs in obstacles:
            dx = current_pos[0] - obs[0]  # direction away from obstacle
            dy = current_pos[1] - obs[1]
            dist = np.hypot(dx, dy)
            
            if dist < 0.1:
                dist = 0.1  # avoid division by zero
            
            # Only apply repulsion within influence range (Equation 2)
            if dist <= self.apf_rho0:
                # Repulsive force magnitude from gradient
                # F_rep = μ * (1/ρ - 1/ρ₀) * (1/ρ²) * direction_away
                magnitude = self.apf_mu * (1.0/dist - 1.0/self.apf_rho0) * (1.0/(dist**2))
                
                f_rep_x += magnitude * dx
                f_rep_y += magnitude * dy
        
        return (f_rep_x, f_rep_y)

    def compute_combined_force(self, current_pos, goal_pos):
        """
        Compute combined force U = U_att + U_rep (Equation 3).
        Returns the negative gradient (force direction).
        """
        # Get attractive force (toward goal)
        f_att = self.compute_attractive_potential_gradient(current_pos, goal_pos)
        
        # Get repulsive force (away from obstacles)
        f_rep = self.compute_repulsive_potential_gradient(current_pos)
        
        # Combined force
        f_total_x = f_att[0] + f_rep[0]
        f_total_y = f_att[1] + f_rep[1]
        
        # Normalize to get direction
        mag = np.hypot(f_total_x, f_total_y)
        if mag > 0.01:
            return (f_total_x / mag, f_total_y / mag)
        else:
            # If no net force, return direction toward goal
            dx = goal_pos[0] - current_pos[0]
            dy = goal_pos[1] - current_pos[1]
            dist = np.hypot(dx, dy)
            if dist > 0.01:
                return (dx / dist, dy / dist)
            return (0.0, 0.0)

    def apply_apf_bias_to_sampling(self, rand_point, nearest_node, goal_pos):
        """
        Apply APF to bias the sampling point (Section 3.1, Figure 4).
        
        Per paper: "The artificial potential field method is first used to 
        improve the sampling points; then, the nodes in the path are 
        re-selected as parents to further optimize the path."
        
        Process (from Figure 4):
        - P1_init: current nearest node
        - P1_rand: random sampling point
        - P1_APF: APF-biased direction from P1_init
        - P1_step: one step toward P1_rand
        - P1_real: actual extended node (blend of APF and random)
        """
        # Compute APF direction from nearest node position
        apf_direction = self.compute_combined_force(
            (nearest_node.x, nearest_node.y), 
            goal_pos
        )
        
        # Compute direction to random point
        dx_rand = rand_point[0] - nearest_node.x
        dy_rand = rand_point[1] - nearest_node.y
        dist_rand = np.hypot(dx_rand, dy_rand)
        
        if dist_rand > 0.01:
            rand_direction = (dx_rand / dist_rand, dy_rand / dist_rand)
        else:
            rand_direction = apf_direction
        
        # Blend APF direction with random direction
        # This creates P1_real from Figure 4
        alpha = 0.5  # weight for APF (can be tuned)
        combined_dx = alpha * apf_direction[0] + (1 - alpha) * rand_direction[0]
        combined_dy = alpha * apf_direction[1] + (1 - alpha) * rand_direction[1]
        
        # Normalize
        combined_mag = np.hypot(combined_dx, combined_dy)
        if combined_mag > 0.01:
            combined_dx /= combined_mag
            combined_dy /= combined_mag
        
        # Apply to get biased sampling point
        biased_x = nearest_node.x + combined_dx * self.step_size * 2
        biased_y = nearest_node.y + combined_dy * self.step_size * 2
        
        # Clip to bounds
        biased_x = np.clip(biased_x, 0, self.env.width - 1)
        biased_y = np.clip(biased_y, 0, self.env.height - 1)
        
        return (biased_x, biased_y)

    # --------------------------
    # RRT* Core Methods
    # --------------------------

    def choose_parent(self, tree: List[Node], new_pos, near_indices):
        """Choose the best parent from nearby nodes based on minimum cost."""
        if not near_indices:
            return None, float('inf')
        
        best_parent = None
        min_cost = float('inf')
        
        for idx in near_indices:
            node = tree[idx]
            edge_cost = self.distance((node.x, node.y), new_pos)
            total_cost = node.cost + edge_cost
            
            if total_cost < min_cost:
                if self.env.is_straight_collision_free((node.x, node.y), new_pos):
                    best_parent = idx
                    min_cost = total_cost
        
        return best_parent, min_cost

    def rewire(self, tree: List[Node], new_idx: int, near_indices, edge_list):
        """Rewire nearby nodes if routing through new node gives lower cost."""
        new_node = tree[new_idx]
        
        for idx in near_indices:
            if idx == new_idx:
                continue
            
            node = tree[idx]
            edge_cost = self.distance((new_node.x, new_node.y), (node.x, node.y))
            new_cost = new_node.cost + edge_cost
            
            if new_cost < node.cost:
                if self.env.is_straight_collision_free((new_node.x, new_node.y),
                                                       (node.x, node.y)):
                    # Remove old edge
                    if node.parent is not None:
                        old_parent = tree[node.parent]
                        old_edge = ((old_parent.x, old_parent.y), (node.x, node.y))
                        if old_edge in edge_list:
                            edge_list.remove(old_edge)
                    
                    # Update parent and cost
                    node.parent = new_idx
                    node.cost = new_cost
                    
                    # Add new edge
                    edge_list.append(((new_node.x, new_node.y), (node.x, node.y)))
                    
                    # Propagate cost updates
                    self.update_descendant_costs(tree, idx)

    def update_descendant_costs(self, tree: List[Node], parent_idx: int):
        """Recursively update costs of all descendants after rewiring."""
        for i, node in enumerate(tree):
            if node.parent == parent_idx:
                edge_cost = self.distance((tree[parent_idx].x, tree[parent_idx].y),
                                        (node.x, node.y))
                node.cost = tree[parent_idx].cost + edge_cost
                self.update_descendant_costs(tree, i)

    # --------------------------
    # Improved Extension Methods
    # --------------------------

    def extend_tree_star(self, tree: List[Node], target_point, edge_list, 
                        use_apf: bool, goal_pos: Tuple[float, float]):
        """
        RRT* extension with optional APF bias.
        
        When use_apf=True: Apply APF bias (for start tree, Section 3.1)
        When use_apf=False: Standard RRT* extension (for goal tree)
        """
        # Find nearest node
        nearest_idx = self.nearest_node_index(tree, target_point)
        nearest_node = tree[nearest_idx]

        # Apply APF bias if this is the start tree
        if use_apf:
            biased_target = self.apply_apf_bias_to_sampling(target_point, nearest_node, goal_pos)
        else:
            biased_target = target_point
        
        # Steer toward (potentially biased) target
        new_x, new_y = self.steer(nearest_node, biased_target)

        # Check collision for new point
        if not self.env.is_free(new_x, new_y):
            return None

        # Find near nodes
        near_indices = self.near_nodes(tree, (new_x, new_y), self.search_radius)
        
        # Choose best parent
        best_parent_idx, min_cost = self.choose_parent(tree, (new_x, new_y), near_indices)
        
        if best_parent_idx is None:
            # Fallback to nearest
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
        
        # Store edge
        edge_list.append(((parent_node.x, parent_node.y), (new_x, new_y)))
        
        # Rewire nearby nodes
        self.rewire(tree, new_index, near_indices, edge_list)
        
        return new_index

    def connect_trees(self, tree1: List[Node], tree2: List[Node], 
                     idx1: int, idx2: int):
        """Check if two nodes from different trees can be connected."""
        node1 = tree1[idx1]
        node2 = tree2[idx2]
        
        dist = self.distance((node1.x, node1.y), (node2.x, node2.y))
        
        if dist <= self.connection_threshold:
            if self.env.is_straight_collision_free((node1.x, node1.y), 
                                         (node2.x, node2.y)):
                return True
        return False

    # --------------------------
    # Main Search
    # --------------------------

    def search(self):
        for it in range(self.max_iters):
            # Sample random point
            rand_point = self.sample_point()

            # Determine goal position based on which tree is the start tree
            if self.start_tree_is_from_start:
                goal_for_apf = self.goal
            else:
                goal_for_apf = self.start

            # Extend start tree WITH APF bias (Section 3.1)
            new_start_idx = self.extend_tree_star(
                self.start_tree, 
                rand_point, 
                self.start_edges,
                use_apf=True,  # APF bias for start tree
                goal_pos=goal_for_apf
            )
            
            if new_start_idx is not None:
                # Get actual position of new node
                new_start_node = self.start_tree[new_start_idx]
                new_start_pos = (new_start_node.x, new_start_node.y)
                
                # Extend goal tree biased toward start tree's new node (Section 3.2)
                # This is the key: goal tree extends toward where start tree just grew
                new_goal_idx = self.extend_tree_star(
                    self.goal_tree, 
                    new_start_pos,  # Target the start tree's new position
                    self.goal_edges,
                    use_apf=False,  # No APF for goal tree, just direct steering
                    goal_pos=goal_for_apf
                )
                
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

            # Swap trees to alternate which tree uses APF
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
            self.start_edges, self.goal_edges = self.goal_edges, self.start_edges
            self.start_tree_is_from_start = not self.start_tree_is_from_start

        print("Goal NOT reached")
        return None

    # --------------------------
    # Path Reconstruction
    # --------------------------

    def reconstruct_path(self, start_idx: int, goal_idx: int):
        """Reconstruct path by tracing back through both trees."""
        # Build path from start tree
        path_from_start = []
        idx = start_idx
        while idx is not None:
            node = self.start_tree[idx]
            path_from_start.append((node.x, node.y))
            idx = node.parent
        
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
            self.final_path = path_from_start + path_from_goal
        else:
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
            ax.plot(px, py, 'y-', linewidth=3, label="Improved Bi-RRT* Path")
            ax.plot(px, py, 'yo', markersize=4)

        # Start/Goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=14, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=18, label='Goal')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title("Improved Bidirectional RRT* with APF")
        ax.legend()
        plt.show()