import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# ============================================
# ROOM ENVIRONMENT
# ============================================

class RoomEnvironment:
    """Room environment with multiple rooms connected by doorways"""
    
    def __init__(self, width: int = 50, height: int = 50, 
                 num_rooms: int = 4, wall_size: int = 2, seed: int = None):
        """
        Create a room environment with walls and doorways
        
        Parameters:
        -----------
        width : int
            Width of the environment grid
        height : int
            Height of the environment grid
        num_rooms : int
            Number of rooms (will be arranged in a grid pattern)
        wall_size : int
            Thickness of the walls (obstacles)
        seed : int
            Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.num_rooms = num_rooms
        self.wall_size = wall_size
        self.seed = seed
        
        # Grid: 0 = free, 1 = obstacle
        self.grid = np.zeros((height, width), dtype=int)
        self._generate_environment()
    
    def _generate_environment(self):
        """Generate room environment with walls and doorways"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Determine grid layout for rooms
        # For num_rooms, create a roughly square grid
        rows = int(np.sqrt(self.num_rooms))
        cols = int(np.ceil(self.num_rooms / rows))
        
        print(f"Creating {rows}x{cols} grid of rooms (total: {rows * cols} rooms)")
        
        # Create walls in a grid pattern
        room_height = self.height // rows
        room_width = self.width // cols
        
        # Draw horizontal walls
        for i in range(1, rows):
            y_pos = i * room_height
            self._add_horizontal_wall(y_pos)
        
        # Draw vertical walls
        for j in range(1, cols):
            x_pos = j * room_width
            self._add_vertical_wall(x_pos)
        
        # Add doorways in walls
        # Horizontal walls - doorways
        for i in range(1, rows):
            y_pos = i * room_height
            for j in range(cols):
                # Add doorway in each wall segment
                door_x = j * room_width + room_width // 2
                door_width = max(3, room_width // 5)  # Doorway width
                self._add_horizontal_doorway(y_pos, door_x, door_width)
        
        # Vertical walls - doorways
        for j in range(1, cols):
            x_pos = j * room_width
            for i in range(rows):
                # Add doorway in each wall segment
                door_y = i * room_height + room_height // 2
                door_width = max(3, room_height // 5)  # Doorway width
                self._add_vertical_doorway(x_pos, door_y, door_width)
        
        # Add outer boundary walls
        self._add_boundary_walls()
        
        # Calculate actual obstacle density
        obstacle_density = np.sum(self.grid == 1) / self.grid.size * 100
        print(f"Room environment generated: {rows * cols} rooms, wall thickness={self.wall_size}, {obstacle_density:.1f}% obstacles")
    
    def _add_horizontal_wall(self, y_center: int):
        """Add a horizontal wall at given y position"""
        half_wall = self.wall_size // 2
        y_start = max(0, y_center - half_wall)
        y_end = min(self.height, y_center + half_wall + 1)
        self.grid[y_start:y_end, :] = 1
    
    def _add_vertical_wall(self, x_center: int):
        """Add a vertical wall at given x position"""
        half_wall = self.wall_size // 2
        x_start = max(0, x_center - half_wall)
        x_end = min(self.width, x_center + half_wall + 1)
        self.grid[:, x_start:x_end] = 1
    
    def _add_horizontal_doorway(self, y_center: int, door_x: int, door_width: int):
        """Cut a doorway through a horizontal wall"""
        half_wall = self.wall_size // 2
        half_door = door_width // 2
        
        y_start = max(0, y_center - half_wall)
        y_end = min(self.height, y_center + half_wall + 1)
        
        x_start = max(0, door_x - half_door)
        x_end = min(self.width, door_x + half_door + 1)
        
        self.grid[y_start:y_end, x_start:x_end] = 0
    
    def _add_vertical_doorway(self, x_center: int, door_y: int, door_width: int):
        """Cut a doorway through a vertical wall"""
        half_wall = self.wall_size // 2
        half_door = door_width // 2
        
        x_start = max(0, x_center - half_wall)
        x_end = min(self.width, x_center + half_wall + 1)
        
        y_start = max(0, door_y - half_door)
        y_end = min(self.height, door_y + half_door + 1)
        
        self.grid[y_start:y_end, x_start:x_end] = 0
    
    def _add_boundary_walls(self):
        """Add walls around the outer boundary"""
        # Top and bottom walls
        self.grid[:self.wall_size, :] = 1
        self.grid[-self.wall_size:, :] = 1
        
        # Left and right walls
        self.grid[:, :self.wall_size] = 1
        self.grid[:, -self.wall_size:] = 1
    
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
        obstacle_density = np.sum(self.grid == 1) / self.grid.size * 100
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.grid, cmap='binary', origin='lower', 
                 extent=[0, self.width, 0, self.height])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Room Environment ({self.num_rooms} rooms, wall thickness={self.wall_size}, {obstacle_density:.1f}% obstacles)')
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
    # Create room environment with 4 rooms
    print("\n1. Creating room environment (4 rooms)...")
    env = RoomEnvironment(width=50, height=50, num_rooms=4, wall_size=2, seed=42)
    print(f"   Environment created: {env.width}x{env.height}")
    
    # Visualize
    print("\n2. Visualizing environment...")
    env.visualize()
    
    # Example with 9 rooms
    print("\n3. Creating room environment (9 rooms)...")
    env_9 = RoomEnvironment(width=50, height=50, num_rooms=9, wall_size=2, seed=123)
    env_9.visualize()
    
    # Example with thick walls
    print("\n4. Creating room environment with thick walls...")
    env_thick = RoomEnvironment(width=50, height=50, num_rooms=4, wall_size=4, seed=456)
    env_thick.visualize()
    
    # Example with thin walls
    print("\n5. Creating room environment with thin walls...")
    env_thin = RoomEnvironment(width=50, height=50, num_rooms=6, wall_size=1, seed=789)
    env_thin.visualize()