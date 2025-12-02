import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set
from bresenham import bresenham
import dubins
from dataclasses import dataclass

# ============================================
# RANDOM ENVIRONMENT
# ============================================

class RandomEnvironment:
    """Simple Random environment for path planning"""
    
    def __init__(self, width: int = 50, height: int = 50, density: int = 30, seed: int = None, robot_radius: float = 2.0):
        """
        Create a random environment with obstacles
        
        Parameters:
        -----------
        width : int
            Width of the environment grid
        height : int
            Height of the environment grid
        density : int
            Obstacle density percentage (0-100)
        seed : int
            Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.density = density
        self.seed = seed
        self.robot_radius = robot_radius
        
        # Grid: 0 = free, 1 = obstacle
        self.grid = np.zeros((height, width), dtype=int)
        self._generate_environment()
    
    def _generate_environment(self):
        """Generate random obstacles based on density"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Generate random obstacles
        probability = self.density / 100.0
        random_grid = np.random.random((self.height, self.width))
        self.grid = (random_grid < probability).astype(int)
        
        # Calculate actual density
        actual_density = np.sum(self.grid == 1) / self.grid.size * 100
        print(f"Environment generated: Target density={self.density}%, Actual density={actual_density:.1f}%")
    
    def is_free(self, x: float, y: float) -> bool:
        """
        Check if a point is free (not an obstacle), considering robot radius
        
        Parameters:
        -----------
        x, y : float
            Coordinates to check
            
        Returns:
        --------
        bool : True if free, False otherwise
        """
        x_int = int(round(x))
        y_int = int(round(y))

        # Check bounds for center point
        if x_int < 0 or x_int >= self.width or y_int < 0 or y_int >= self.height:
            return False

        # Check all cells within robot radius
        radius_cells = int(np.ceil(self.robot_radius))

        # Define bounds once
        min_x = max(0,           int(round(x - radius_cells)))
        max_x = min(self.width,  int(round(x + radius_cells + 1)))
        min_y = max(0,           int(round(y - radius_cells)))
        max_y = min(self.height, int(round(y + radius_cells + 1)))

        # Pre-compute radius squared to avoid sqrt in loop
        radius_sq = self.robot_radius ** 2

        # Check region using array slicing
        for dy in range(min_y - y_int, max_y - y_int):
            for dx in range(min_x - x_int, max_x - x_int):
                # Use squared distance to avoid sqrt
                if dx*dx + dy*dy <= radius_sq:
                    check_x = x_int + dx
                    check_y = y_int + dy
                    
                    # Check if obstacle
                    if self.grid[check_y, check_x] == 1:
                        return False

        return True
 
    def is_straight_collision_free(self, pos_a: Tuple[float, float], pos_b: Tuple[float, float], method: str = '') -> bool:
        """
        Check if line segment between two points is collision-free using grid traversal
        
        Parameters:
        -----------
        pos_a : tuple (x, y)
            Start position
        pos_b : tuple (x, y)
            End position
        method : str
            'bresenham' - standard Bresenham (faster, may miss some cells)
            
        Returns:
        --------
        bool : True if line is collision-free, False otherwise
        """
        x0, y0 = pos_a
        x1, y1 = pos_b
        
        # Get all cells crossed by the line
        if method == 'bresenham':
            # Convert to grid coordinates
            x0_grid = int(round(x0))
            y0_grid = int(round(y0))
            x1_grid = int(round(x1))
            y1_grid = int(round(y1))
            cells = bresenham(x0_grid, y0_grid, x1_grid, y1_grid)

            # Check if any cell is an obstacle or out of bounds
            for x, y in cells:
                if x < 0 or x >= self.width or y < 0 or y >= self.height or self.grid[y, x] == 1:
                    return False
            return True
        
        else: 
            # Calculate distance and number of checks
            distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            num_checks = max(int(distance * 2), 10)
            
            # Check points along the line
            for i in range(num_checks + 1):
                t = i / num_checks
                x = x1 + t * (x1 - x0)
                y = y1 + t * (y1 - y0)
                
                if not self.is_free(x, y):
                    return False
            
        return True
        
    def is_dubins_collision_free(self, from_node, to_node, turning_radius):
        q0 = (from_node.x, from_node.y, from_node.theta)
        q1 = (to_node.x, to_node.y, to_node.theta)
        path = dubins.shortest_path(q0, q1, turning_radius)
        
        # Sample along Dubins path
        sample_distance = 0.5
        for i in np.arange(0, path.path_length(), sample_distance):
            x, y, _ = path.sample(i)
            if not self.is_free(x, y):
                return False
        return True
    
    def sample_free_point(self, max_attempts=1000) -> Tuple[float, float]:
        """Sample a random free point in the environment"""
        for _ in range(max_attempts):
            x = np.random.uniform(0+self.robot_radius, self.width-self.robot_radius)
            y = np.random.uniform(0+self.robot_radius, self.height-self.robot_radius)
            if self.is_free(x, y):
                return (x, y)
        
        # Fallback: find any free cell
        free_cells = np.argwhere(self.grid == 0)
        if len(free_cells) > 0:
            idx = np.random.randint(len(free_cells))
            y, x = free_cells[idx]
            return (float(x), float(y))
        
        raise RuntimeError("No free space in environment")
    
    def visualize(self):
        """Visualize the environment"""
        actual_density = np.sum(self.grid == 1) / self.grid.size * 100
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.grid, cmap='binary', origin='lower', 
                 extent=[0, self.width, 0, self.height])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Random Environment (Target: {self.density}%, Actual: {actual_density:.1f}%)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def show_path(self, 
                  start: Tuple[float, float],
                  goal: Tuple[float, float],
                  path: Optional[List[Tuple[float, float]]] = None):
        """
        Visualize the environment with start, goal, and path
        
        Parameters:
        -----------
        start : tuple (x, y)
            Start position
        goal : tuple (x, y)
            Goal position
        path : list of tuples, optional
            Path as [(x, y), ...] waypoints
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw environment
        ax.imshow(self.grid, cmap='binary', origin='lower',
                 extent=[0, self.width, 0, self.height])
        
        # Draw path (if found)
        if path and len(path) > 0:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path', alpha=0.8)
            
            # Add waypoint markers
            ax.plot(path_x, path_y, 'bo', markersize=6, alpha=0.6)
        
        # Draw start (green circle)
        ax.plot(start[0], start[1], 'go', markersize=15,
               label='Start', markeredgecolor='black', markeredgewidth=2)
        
        # Draw goal (red star)
        ax.plot(goal[0], goal[1], 'r*', markersize=20,
               label='Goal', markeredgecolor='black', markeredgewidth=2)
        
        # Configure plot
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # Simple title
        if path:
            path_length = sum(np.sqrt((path[i+1][0] - path[i][0])**2 + 
                                     (path[i+1][1] - path[i][1])**2)
                            for i in range(len(path) - 1))
            title = f'Path Planning - Path Found (Length: {path_length:.2f})'
        else:
            title = 'Path Planning - No Path Found'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Create random environment
    print("\n1. Creating random environment...")
    env = RandomEnvironment(width=50, height=50, density=10, seed=42)
    print(f"   Environment created: {env.width}x{env.height}, density={env.density}%")
    
    # Visualize just the environment
    print("\n2. Visualizing environment...")
    env.visualize()