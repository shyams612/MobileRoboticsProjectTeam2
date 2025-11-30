import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ============================================
# RANDOM ENVIRONMENT
# ============================================

class RandomEnvironment:
    """Simple Random environment for path planning"""
    
    def __init__(self, width: int = 50, height: int = 50, density: int = 30, seed: int = None):
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
        Check if a point is free (not an obstacle)
        
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
        
        if x_int < 0 or x_int >= self.width or y_int < 0 or y_int >= self.height:
            return False
        
        return self.grid[y_int, x_int] == 0
    
    def is_collision_free(self, pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> bool:
        """
        Check if line segment between two points is collision-free
        
        Parameters:
        -----------
        pos_a : tuple (x, y)
            Start position
        pos_b : tuple (x, y)
            End position
            
        Returns:
        --------
        bool : True if line is collision-free, False otherwise
        """
        x1, y1 = pos_a
        x2, y2 = pos_b
        
        # Calculate distance and number of checks
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_checks = max(int(distance * 2), 10)
        
        # Check points along the line
        for i in range(num_checks + 1):
            t = i / num_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            if not self.is_free(x, y):
                return False
        
        return True
    
    def sample_free_point(self, max_attempts=1000) -> Tuple[float, float]:
        """Sample a random free point in the environment"""
        for _ in range(max_attempts):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
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
    