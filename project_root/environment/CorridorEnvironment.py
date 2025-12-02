import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# ============================================
# CORRIDOR ENVIRONMENT
# ============================================

class CorridorEnvironment:
    """Corridor environment with walls and passages for path planning"""
    
    def __init__(self, width: int = 50, height: int = 50, 
                 corridor_width: int = 5, num_corridors: int = 3, seed: int = None):
        """
        Create a corridor environment with passages
        
        Parameters:
        -----------
        width : int
            Width of the environment grid
        height : int
            Height of the environment grid
        corridor_width : int
            Width of the corridors (passages)
        num_corridors : int
            Number of corridor segments to create
        seed : int
            Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.corridor_width = corridor_width
        self.num_corridors = num_corridors
        self.seed = seed
        
        # Grid: 0 = free, 1 = obstacle
        self.grid = np.ones((height, width), dtype=int)  # Start with all obstacles
        self._generate_environment()
    
    def _generate_environment(self):
        """Generate corridor environment with walls and passages"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Create horizontal and vertical corridors
        for i in range(self.num_corridors):
            if i % 2 == 0:
                # Horizontal corridor
                y_pos = np.random.randint(self.corridor_width, self.height - self.corridor_width)
                self._add_horizontal_corridor(y_pos)
            else:
                # Vertical corridor
                x_pos = np.random.randint(self.corridor_width, self.width - self.corridor_width)
                self._add_vertical_corridor(x_pos)
        
        # Add some open areas at intersections or random locations
        num_open_areas = max(2, self.num_corridors // 2)
        for _ in range(num_open_areas):
            x_center = np.random.randint(self.corridor_width * 2, self.width - self.corridor_width * 2)
            y_center = np.random.randint(self.corridor_width * 2, self.height - self.corridor_width * 2)
            size = self.corridor_width * 2
            self._add_open_area(x_center, y_center, size)
        
        # Calculate actual free space
        free_space = np.sum(self.grid == 0) / self.grid.size * 100
        print(f"Corridor environment generated: {self.num_corridors} corridors, {free_space:.1f}% free space")
    
    def _add_horizontal_corridor(self, y_center: int):
        """Add a horizontal corridor at given y position"""
        half_width = self.corridor_width // 2
        y_start = max(0, y_center - half_width)
        y_end = min(self.height, y_center + half_width + 1)
        self.grid[y_start:y_end, :] = 0
    
    def _add_vertical_corridor(self, x_center: int):
        """Add a vertical corridor at given x position"""
        half_width = self.corridor_width // 2
        x_start = max(0, x_center - half_width)
        x_end = min(self.width, x_center + half_width + 1)
        self.grid[:, x_start:x_end] = 0
    
    def _add_open_area(self, x_center: int, y_center: int, size: int):
        """Add an open area (room) at given position"""
        half_size = size // 2
        x_start = max(0, x_center - half_size)
        x_end = min(self.width, x_center + half_size)
        y_start = max(0, y_center - half_size)
        y_end = min(self.height, y_center + half_size)
        self.grid[y_start:y_end, x_start:x_end] = 0
    
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
            x = np.random.random() * self.width
            y = np.random.random() * self.height
            if self.is_free(x, y):
                return (x, y)
        
        print("Falling back to exhaustive search for free point")
        # Fallback: find any free cell
        free_cells = np.argwhere(self.grid == 0)
        if len(free_cells) > 0:
            idx = np.random.randint(len(free_cells))
            y, x = free_cells[idx]
            return (float(x), float(y))
        
        raise RuntimeError("No free space in environment")
    
    def visualize(self):
        """Visualize the environment"""
        free_space = np.sum(self.grid == 0) / self.grid.size * 100
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.grid, cmap='binary', origin='lower', 
                 extent=[0, self.width, 0, self.height])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Corridor Environment ({self.num_corridors} corridors, {free_space:.1f}% free space)')
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
    # Create corridor environment
    print("\n1. Creating corridor environment...")
    env = CorridorEnvironment(width=50, height=50, corridor_width=5, num_corridors=4, seed=42)
    print(f"   Environment created: {env.width}x{env.height}")
    
    # Visualize just the environment
    print("\n2. Visualizing environment...")
    env.visualize()
    
    # Example with different configurations
    print("\n3. Creating narrow corridor environment...")
    env_narrow = CorridorEnvironment(width=50, height=50, corridor_width=3, num_corridors=5, seed=123)
    env_narrow.visualize()
    
    print("\n4. Creating wide corridor environment...")
    env_wide = CorridorEnvironment(width=50, height=50, corridor_width=8, num_corridors=3, seed=456)
    env_wide.visualize()