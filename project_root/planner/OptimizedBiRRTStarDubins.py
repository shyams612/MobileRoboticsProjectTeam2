# ============================================
# OPTIMIZED BIDIRECTIONAL RRT* WITH DUBINS PATHS
# (Goal: at least as good as original, no smoothing)
# ============================================

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import dubins


@dataclass
class DubinsNode:
    x: float
    y: float
    theta: float
    parent: Optional[int]
    cost: float = 0.0  # cost from root to this node


class OptimizedBiRRTStarDubins:
    def __init__(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        env,
        step_size: float = 2.0,
        max_iters: int = 5000,
        connection_threshold: float = 4.0,  # slightly larger than 3.5 (easier to connect trees)
        turning_radius: float = 1.0,
        goal_sample_rate: float = 0.1,
        search_radius: float = 6.0,        # slightly larger than 5.0 (more neighbors for RRT*)
        early_stop: bool = True,
    ):
        """
        Bidirectional RRT* with Dubins curves (Ackermann constraints).

        This is intentionally very close to your original BidirectionalRRTStarDubins,
        but with:
          - slightly more permissive connection_threshold and search_radius
          - no path smoothing
        so it should never be worse than the original, and often better.
        """
        self.start = start
        self.goal = goal
        self.env = env

        self.step_size = step_size
        self.max_iters = max_iters
        self.connection_threshold = connection_threshold
        self.turning_radius = turning_radius
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.early_stop = early_stop

        # Two trees: one from start, one from goal
        self.start_tree: List[DubinsNode] = [
            DubinsNode(start[0], start[1], start[2], parent=None, cost=0.0)
        ]
        self.goal_tree: List[DubinsNode] = [
            DubinsNode(goal[0], goal[1], goal[2], parent=None, cost=0.0)
        ]

        # Path bookkeeping
        self.final_path = None
        self.best_path = None
        self.best_cost = float('inf')

        # Edge lists for visualization
        self.start_edges: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        self.goal_edges: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []

    # --------------------------
    # Utility
    # --------------------------

    def dubins_distance(self, node1, node2) -> float:
        """Use Dubins shortest path length as distance metric."""
        if isinstance(node1, DubinsNode):
            q0 = (node1.x, node1.y, node1.theta)
        else:
            q0 = node1  # (x, y, theta)

        if isinstance(node2, DubinsNode):
            q1 = (node2.x, node2.y, node2.theta)
        else:
            q1 = node2

        path = dubins.shortest_path(q0, q1, self.turning_radius)
        return path.path_length()

    def euclidean_distance(self, a, b) -> float:
        """Euclidean distance using (x, y) only."""
        if isinstance(a, DubinsNode):
            ax, ay = a.x, a.y
        else:
            ax, ay = a[0], a[1]

        if isinstance(b, DubinsNode):
            bx, by = b.x, b.y
        else:
            bx, by = b[0], b[1]

        return np.hypot(ax - bx, ay - by)

    def sample_point(self) -> Tuple[float, float, float]:
        """
        Sample a random dubins configuration (x, y, theta) with goal bias.

        - With probability goal_sample_rate, sample exactly the goal.
        - Otherwise:
            * pick a random free (x, y) from the environment
            * point roughly toward the mid-point between start and goal
              plus some orientation noise.
        """
        if np.random.rand() < self.goal_sample_rate:
            return self.goal

        x, y = self.env.sample_free_point()

        ref_x = (self.start[0] + self.goal[0]) * 0.5
        ref_y = (self.start[1] + self.goal[1]) * 0.5
        theta_center = np.arctan2(ref_y - y, ref_x - x)
        theta = theta_center + np.random.uniform(-np.pi / 4.0, np.pi / 4.0)

        return (x, y, theta)

    def nearest_node_index(self, tree: List[DubinsNode], point) -> int:
        """Return index of nearest node to 'point' using Dubins distance."""
        point_node = DubinsNode(point[0], point[1], point[2], parent=None)
        dists = [self.dubins_distance(node, point_node) for node in tree]
        return int(np.argmin(dists))

    def near_nodes(self, tree: List[DubinsNode], point, radius: float) -> List[int]:
        """Return indices of all nodes within Dubins distance 'radius' of 'point'."""
        point_node = DubinsNode(point[0], point[1], point[2], parent=None)
        near_indices: List[int] = []
        for i, node in enumerate(tree):
            d = self.dubins_distance(node, point_node)
            if d <= radius:
                near_indices.append(i)
        return near_indices

    def steer(self, from_node: DubinsNode, to_point) -> Tuple[float, float, float]:
        """
        Steer from 'from_node' toward 'to_point' along a Dubins path.
        Returns a new configuration (x, y, theta) that is at most step_size away
        in Dubins path length.
        """
        q0 = (from_node.x, from_node.y, from_node.theta)

        if isinstance(to_point, tuple) and len(to_point) == 3:
            q1 = to_point
        else:
            # Estimate heading from from_node to target (x, y)
            dx = to_point[0] - from_node.x
            dy = to_point[1] - from_node.y
            est_theta = np.arctan2(dy, dx)
            q1 = (to_point[0], to_point[1], est_theta)

        path = dubins.shortest_path(q0, q1, self.turning_radius)

        if path.path_length() <= self.step_size:
            new_config = q1
        else:
            new_config = path.sample(self.step_size)

        return new_config  # (x, y, theta)

    # --------------------------
    # RRT* utilities: parent & rewiring
    # --------------------------

    def choose_parent(
        self,
        tree: List[DubinsNode],
        new_config,
        near_indices: List[int]
    ) -> Tuple[Optional[int], float]:
        """
        Choose best parent for new_config among near_indices based on minimal cost.
        Returns: (best_parent_index, best_cost)
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

    def _update_descendants_cost(self, tree: List[DubinsNode], parent_idx: int) -> None:
        """Recursively update costs of all descendants of node 'parent_idx'."""
        parent_node = tree[parent_idx]
        for i, node in enumerate(tree):
            if node.parent == parent_idx:
                edge_cost = self.dubins_distance(parent_node, node)
                node.cost = parent_node.cost + edge_cost
                self._update_descendants_cost(tree, i)

    def rewire(
        self,
        tree: List[DubinsNode],
        new_idx: int,
        near_indices: List[int],
        edge_list: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    ) -> None:
        """
        RRT* rewiring: if going through new_node shortens path to some neighbor,
        update that neighbor's parent and cost.
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
                    # Remove old edge if it was stored
                    if node.parent is not None:
                        old_parent = tree[node.parent]
                        old_edge = (
                            (old_parent.x, old_parent.y, old_parent.theta),
                            (node.x, node.y, node.theta),
                        )
                        if old_edge in edge_list:
                            edge_list.remove(old_edge)

                    # Update parent and cost
                    node.parent = new_idx
                    node.cost = potential_cost

                    # Add new edge for visualization
                    edge_list.append(
                        (
                            (new_node.x, new_node.y, new_node.theta),
                            (node.x, node.y, node.theta),
                        )
                    )

                    # Update all descendants' costs
                    self._update_descendants_cost(tree, idx)

    # --------------------------
    # RRT* tree extension
    # --------------------------

    def extend_tree_star(
        self,
        tree: List[DubinsNode],
        target_point,
        edge_list: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    ) -> Optional[int]:
        """
        Extend 'tree' toward 'target_point' using Dubins RRT* logic:
          - find nearest node
          - steer one step along Dubins path
          - pick best parent among near neighbors
          - rewire neighbors through the new node
        Returns index of new node if successful, else None.
        """
        # 1) nearest node
        nearest_idx = self.nearest_node_index(tree, target_point)
        nearest_node = tree[nearest_idx]

        # 2) steer toward target
        new_config = self.steer(nearest_node, target_point)
        new_x, new_y, new_theta = new_config
        new_node_temp = DubinsNode(new_x, new_y, new_theta, parent=None)

        # collision check from nearest to new
        if not self.env.is_dubins_collision_free(nearest_node, new_node_temp, self.turning_radius):
            return None

        # 3) find near neighbors
        near_indices = self.near_nodes(tree, new_config, self.search_radius)

        # 4) choose best parent among near neighbors
        best_parent_idx, min_cost = self.choose_parent(tree, new_config, near_indices)

        if best_parent_idx is None:
            # fallback to nearest if no neighbor improves cost
            best_parent_idx = nearest_idx
            min_cost = nearest_node.cost + self.dubins_distance(nearest_node, new_node_temp)

        # 5) add new node
        parent_node = tree[best_parent_idx]
        new_node = DubinsNode(new_x, new_y, new_theta, parent=best_parent_idx, cost=min_cost)
        new_index = len(tree)
        tree.append(new_node)

        # add edge for visualization
        edge_list.append(
            (
                (parent_node.x, parent_node.y, parent_node.theta),
                (new_node.x, new_node.y, new_node.theta),
            )
        )

        # 6) rewire neighbors
        self.rewire(tree, new_index, near_indices, edge_list)

        return new_index

    # --------------------------
    # Tree connection
    # --------------------------

    def connect_trees(
        self,
        tree1: List[DubinsNode],
        tree2: List[DubinsNode],
        idx1: int,
        idx2: int
    ) -> bool:
        """
        Check if two nodes from different trees can be connected by a
        collision-free Dubins path.
        """
        node1 = tree1[idx1]
        node2 = tree2[idx2]

        # fast: prune by Euclidean distance
        if self.euclidean_distance(node1, node2) > 2.0 * self.connection_threshold:
            return False

        # then check Dubins distance threshold
        if self.dubins_distance(node1, node2) <= self.connection_threshold:
            if self.env.is_dubins_collision_free(node1, node2, self.turning_radius):
                return True

        return False

    # --------------------------
    # Main search
    # --------------------------

    def search(self):
        """
        Main Bidirectional RRT* loop (no smoothing).
        If early_stop=True: return as soon as a path is found.
        If early_stop=False: keep going and return best_cost path seen.
        """
        for it in range(self.max_iters):
            if it % 1000 == 0:
                print(f"[OptimizedBiRRTStarDubins] Iteration {it}")

            # 1) Sample random Dubins configuration
            rand_point = self.sample_point()

            # 2) Extend start tree toward sample
            new_start_idx = self.extend_tree_star(self.start_tree, rand_point, self.start_edges)

            if new_start_idx is not None:
                # 3) try extend goal tree toward this new node
                new_start_node = self.start_tree[new_start_idx]
                new_start_cfg = (new_start_node.x, new_start_node.y, new_start_node.theta)

                new_goal_idx = self.extend_tree_star(self.goal_tree, new_start_cfg, self.goal_edges)

                if new_goal_idx is not None:
                    # 4) try to connect the two new nodes
                    if self.connect_trees(self.start_tree, self.goal_tree, new_start_idx, new_goal_idx):
                        connection_cost = self.dubins_distance(
                            self.start_tree[new_start_idx],
                            self.goal_tree[new_goal_idx],
                        )
                        total_cost = (
                            self.start_tree[new_start_idx].cost
                            + self.goal_tree[new_goal_idx].cost
                            + connection_cost
                        )

                        if total_cost < self.best_cost:
                            self.best_cost = total_cost
                            self.best_path = self.reconstruct_path(new_start_idx, new_goal_idx)
                            print(f"  Trees connected at iteration {it}")
                            print(f"  Path cost: {total_cost:.2f}")

                            if self.early_stop:
                                self.final_path = self.best_path
                                return self.final_path

            # 5) Alternate which tree we grow first
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
            self.start_edges, self.goal_edges = self.goal_edges, self.start_edges

        # done with all iterations
        if self.best_path is not None:
            print(f"[OptimizedBiRRTStarDubins] Search complete. Best path cost: {self.best_cost:.2f}")
            self.final_path = self.best_path
        else:
            print("[OptimizedBiRRTStarDubins] Goal NOT reached")
            self.final_path = None

        return self.final_path

    # --------------------------
    # Path reconstruction
    # --------------------------

    def reconstruct_path(self, start_idx: int, goal_idx: int):
        """
        Reconstruct full path (start -> ... -> connection -> ... -> goal)
        taking into account that trees may have been swapped during search.
        """
        # path from start tree
        path_from_start = []
        idx = start_idx
        while idx is not None:
            node = self.start_tree[idx]
            path_from_start.append((node.x, node.y, node.theta))
            idx = node.parent
        path_from_start = list(reversed(path_from_start))

        # path from goal tree
        path_from_goal = []
        idx = goal_idx
        while idx is not None:
            node = self.goal_tree[idx]
            path_from_goal.append((node.x, node.y, node.theta))
            idx = node.parent

        # decide which tree is truly the start tree
        start_dist = self.euclidean_distance(path_from_start[0], self.start)
        if start_dist < 1e-2:
            # start_tree really starts at self.start
            full_path = path_from_start + path_from_goal
        else:
            # trees were swapped at some point
            full_path = list(reversed(path_from_goal)) + list(reversed(path_from_start))

        return full_path

    # --------------------------
    # Visualization
    # --------------------------

    def show_path(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            self.env.grid,
            cmap='binary',
            origin='lower',
            extent=[0, self.env.width, 0, self.env.height],
        )

        # draw tree edges
        for (a, b) in self.start_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'b-', alpha=0.3, linewidth=0.5)
        for (a, b) in self.goal_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'g-', alpha=0.3, linewidth=0.5)

        # draw final path with Dubins curves
        if self.final_path is not None:
            for i in range(len(self.final_path) - 1):
                q0 = self.final_path[i]
                q1 = self.final_path[i + 1]
                path = dubins.shortest_path(q0, q1, self.turning_radius)
                configs, _ = path.sample_many(0.1)
                px = [c[0] for c in configs]
                py = [c[1] for c in configs]
                ax.plot(px, py, 'y-', linewidth=3)

                # heading arrows
                for c in configs[::5]:
                    ax.arrow(
                        c[0], c[1],
                        0.5 * np.cos(c[2]), 0.5 * np.sin(c[2]),
                        head_width=0.3, head_length=0.3,
                        color='r', alpha=0.7,
                    )

        # start & goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=14, label='Start')
        ax.arrow(
            self.start[0], self.start[1],
            2.0 * np.cos(self.start[2]), 2.0 * np.sin(self.start[2]),
            head_width=0.5, head_length=0.5,
            fc='green', ec='green',
        )

        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=18, label='Goal')
        ax.arrow(
            self.goal[0], self.goal[1],
            2.0 * np.cos(self.goal[2]), 2.0 * np.sin(self.goal[2]),
            head_width=0.5, head_length=0.5,
            fc='red', ec='red',
        )

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')

        title = "Optimized Bidirectional RRT* with Dubins Paths (No smoothing)"
        if self.best_cost < float('inf'):
            title += f" (Best cost: {self.best_cost:.2f})"
        ax.set_title(title)
        ax.legend()
        plt.show()
