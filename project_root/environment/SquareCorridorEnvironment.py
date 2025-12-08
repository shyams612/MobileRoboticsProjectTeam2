import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import dubins

# ============================================
# SQUARE CORRIDOR ENVIRONMENT
# ============================================

class SquareCorridorEnvironment:
    """
    Square corridor environment representing a floor layout.
    
    A rectangular corridor goes around a central obstacle/core area,
    like walking around the perimeter of a building floor or hallway system.
    The environment has a central blocked area with a corridor surrounding it.
    """
    
    def __init__(self, width: int = 100, height: int = 100, 
                 corridor_width: int = 5, center_margin: int = 30, seed: int = None, robot_radius: float = 2.0):
        """
        Create a square corridor environment (like a building floor).
        
        Parameters:
        -----------
        width : int
            Width of the environment grid
        height : int
            Height of the environment grid
        corridor_width : int
            Width of the corridor passage
        center_margin : int
            Size of the central obstacle/blocked area (distance from center)
        seed : int
            Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.corridor_width = corridor_width
        self.center_margin = center_margin
        self.seed = seed
        self.robot_radius = robot_radius

        
        # Grid: 0 = free, 1 = obstacle
        self.grid = np.ones((height, width), dtype=int)  # Start with all obstacles
        
        self._generate_environment()
    
    def _generate_environment(self):
        """Generate square corridor environment around a central core"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Create outer boundary with corridor
        self._create_outer_corridor()
        
        # Calculate actual free space
        free_space = np.sum(self.grid == 0) / self.grid.size * 100
        print(f"Square corridor environment generated: {free_space:.1f}% free space (corridor around central core)")
    
    def _create_outer_corridor(self):
        """Create the main square corridor around a central area"""
        center_x = self.width // 2
        center_y = self.height // 2
        # Define inner rectangle (exclusive upper bounds) and outer rectangle
        inner_left = center_x - self.center_margin
        inner_right = center_x + self.center_margin
        inner_bottom = center_y - self.center_margin
        inner_top = center_y + self.center_margin

        # Use slice-style bounds (upper bounds exclusive) and clip to grid limits
        outer_left = max(0, inner_left - self.corridor_width)
        outer_right = min(self.width, inner_right + self.corridor_width)
        outer_bottom = max(0, inner_bottom - self.corridor_width)
        outer_top = min(self.height, inner_top + self.corridor_width)

        # Top corridor: rows from inner_top to outer_top (width = outer_top - inner_top)
        if inner_top < outer_top:
            self.grid[inner_top:outer_top, outer_left:outer_right] = 0

        # Bottom corridor: rows from outer_bottom to inner_bottom (width = inner_bottom - outer_bottom)
        if outer_bottom < inner_bottom:
            self.grid[outer_bottom:inner_bottom, outer_left:outer_right] = 0

        # Left corridor: columns from outer_left to inner_left (width = inner_left - outer_left)
        if outer_left < inner_left:
            self.grid[outer_bottom:outer_top, outer_left:inner_left] = 0

        # Right corridor: columns from inner_right to outer_right (width = outer_right - inner_right)
        if inner_right < outer_right:
            self.grid[outer_bottom:outer_top, inner_right:outer_right] = 0
    
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
        """Sample a random free point in the corridor"""
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
        """Visualize the square corridor environment"""
        free_space = np.sum(self.grid == 0) / self.grid.size * 100
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid: obstacles in black, free space in white
        ax.imshow(self.grid, cmap='binary', origin='lower', 
                 extent=[0, self.width, 0, self.height])
        
        # Add grid lines to show corridor structure
        center_x = self.width // 2
        center_y = self.height // 2
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Square Corridor Environment ({free_space:.1f}% free space)')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.legend(loc='upper right')
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
        
        # Title with path info
        if path:
            path_length = sum(np.sqrt((path[i+1][0] - path[i][0])**2 + 
                                     (path[i+1][1] - path[i][1])**2)
                            for i in range(len(path) - 1))
            title = f'Square Corridor Path Planning - Path Found (Length: {path_length:.2f})'
        else:
            title = 'Square Corridor Path Planning - No Path Found'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

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

    

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Create square corridor environment
    print("\n1. Creating square corridor environment...")
    env = SquareCorridorEnvironment(width=120, height=120, corridor_width=15,
                                    center_margin=35, seed=456)
    print(f"   Environment created: {env.width}x{env.height}")
    
    # Visualize the environment
    print("\n2. Visualizing environment...")
    env.visualize()
    
    # Example with different configurations
    print("\n3. Creating narrow corridor environment...")
    env_narrow = SquareCorridorEnvironment(width=120, height=120, corridor_width=15,
                                           center_margin=35, seed=456)
    env_narrow.visualize()
    
    print("\n4. Creating wide corridor environment...")
    env_wide = SquareCorridorEnvironment(width=120, height=120, corridor_width=15,
                                         center_margin=35, seed=456)
    env_wide.visualize()
