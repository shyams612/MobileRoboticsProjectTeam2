import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# ============================================
# CORRIDOR ENVIRONMENT
# ============================================

class CorridorEnvironment:
    """
    Corridor environment with rooms connected by passages.
    
    Rooms are bounded rectangular spaces, and corridors are continuous passages
    with obstacles (walls) on both sides connecting different rooms.
    """
    
    def __init__(self, width: int = 50, height: int = 50, 
                 corridor_width: int = 5, num_corridors: int = 3, seed: int = None):
        """
        Create a corridor environment with rooms and connecting passages.
        
        Parameters:
        -----------
        width : int
            Width of the environment grid
        height : int
            Height of the environment grid
        corridor_width : int
            Width of the corridors (passages between rooms)
        num_corridors : int
            Number of corridors to create (also determines number of rooms)
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
        self.rooms = []  # List of room rectangles: [(x1, y1, x2, y2), ...]
        
        self._generate_environment()
    
    def _generate_environment(self):
        """Generate corridor environment with rooms and connecting passages"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Step 1: Initialize with obstacles everywhere
        self.grid = np.ones((self.height, self.width), dtype=int)
        
        # Step 2: Create rooms scattered throughout the environment (as free space)
        num_rooms = self.num_corridors + 1
        self._create_rooms(num_rooms)
        
        # Step 3: Create corridors connecting rooms (as free passages)
        self._create_connecting_corridors()
        
        # Calculate actual free space
        free_space = np.sum(self.grid == 0) / self.grid.size * 100
        print(f"Corridor environment generated: {len(self.rooms)} rooms, "
              f"{self.num_corridors} corridors, {free_space:.1f}% free space")
    
    def _create_rooms(self, num_rooms: int):
        """Create random rectangular rooms in the environment (as free space)"""
        min_room_size = max(self.corridor_width * 2, 5)
        max_room_size = max(min(self.corridor_width * 6, self.width // 4), min_room_size + 1)
        
        created_rooms = 0
        for attempt_round in range(5):  # Multiple rounds to create rooms
            for _ in range(num_rooms * 2):  # Extra attempts
                # Random room dimensions
                room_width = np.random.randint(min_room_size, max_room_size + 1)
                room_height = np.random.randint(min_room_size, max_room_size + 1)
                
                # Random position with margins
                margin = self.corridor_width
                
                # Ensure we have valid bounds
                x_min = margin
                x_max = self.width - room_width - margin
                y_min = margin
                y_max = self.height - room_height - margin
                
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                x1 = np.random.randint(x_min, x_max + 1)
                y1 = np.random.randint(y_min, y_max + 1)
                x2 = x1 + room_width
                y2 = y1 + room_height
                
                # Check for overlap with existing rooms
                overlaps = False
                for (rx1, ry1, rx2, ry2) in self.rooms:
                    # Add buffer zone around rooms
                    buffer = self.corridor_width
                    if not (x2 + buffer < rx1 or x1 - buffer > rx2 or 
                            y2 + buffer < ry1 or y1 - buffer > ry2):
                        overlaps = True
                        break
                
                if not overlaps:
                    # Add room and mark as free space (0)
                    self.rooms.append((x1, y1, x2, y2))
                    self.grid[y1:y2, x1:x2] = 0
                    created_rooms += 1
                    
                    if created_rooms >= num_rooms:
                        return
    
    def _create_connecting_corridors(self):
        """Create corridors connecting adjacent rooms"""
        num_rooms = len(self.rooms)
        
        # Create corridors between pairs of rooms
        for i in range(min(self.num_corridors, num_rooms - 1)):
            room_a_idx = i % num_rooms
            room_b_idx = (i + 1) % num_rooms
            
            room_a = self.rooms[room_a_idx]
            room_b = self.rooms[room_b_idx]
            
            # Get center points of rooms
            center_a = ((room_a[0] + room_a[2]) / 2, (room_a[1] + room_a[3]) / 2)
            center_b = ((room_b[0] + room_b[2]) / 2, (room_b[1] + room_b[3]) / 2)
            
            # Create L-shaped corridor (horizontal then vertical, or vice versa)
            if np.random.random() < 0.5:
                # Horizontal first, then vertical
                self._add_horizontal_passage(center_a[1], center_a[0], center_b[0])
                self._add_vertical_passage(center_b[0], center_a[1], center_b[1])
            else:
                # Vertical first, then horizontal
                self._add_vertical_passage(center_a[0], center_a[1], center_b[1])
                self._add_horizontal_passage(center_b[1], center_a[0], center_b[0])
    
    def _add_horizontal_passage(self, y_center: float, x_start: float, x_end: float):
        """
        Add a horizontal corridor (passage) between two x-coordinates.
        
        The corridor is a continuous free space with walls on both sides.
        """
        x_start_int = int(round(min(x_start, x_end)))
        x_end_int = int(round(max(x_start, x_end)))
        y_center_int = int(round(y_center))
        
        # Corridor dimensions
        half_width = self.corridor_width // 2
        y_top = max(0, y_center_int + half_width)
        y_bottom = max(0, y_center_int - half_width)
        
        # Mark corridor as free space
        x_start_int = max(0, x_start_int)
        x_end_int = min(self.width, x_end_int)
        
        self.grid[y_bottom:y_top + 1, x_start_int:x_end_int] = 0
    
    def _add_vertical_passage(self, x_center: float, y_start: float, y_end: float):
        """
        Add a vertical corridor (passage) between two y-coordinates.
        
        The corridor is a continuous free space with walls on both sides.
        """
        y_start_int = int(round(min(y_start, y_end)))
        y_end_int = int(round(max(y_start, y_end)))
        x_center_int = int(round(x_center))
        
        # Corridor dimensions
        half_width = self.corridor_width // 2
        x_right = min(self.width - 1, x_center_int + half_width)
        x_left = max(0, x_center_int - half_width)
        
        # Mark corridor as free space
        y_start_int = max(0, y_start_int)
        y_end_int = min(self.height, y_end_int)
        
        self.grid[y_start_int:y_end_int, x_left:x_right + 1] = 0
    
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
    
    def _is_in_room(self, x: float, y: float) -> bool:
        """Check if a point is inside any defined room"""
        for (x1, y1, x2, y2) in self.rooms:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False
    
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
        """Visualize the environment with rooms and corridors"""
        free_space = np.sum(self.grid == 0) / self.grid.size * 100
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid: obstacles in black, free space in white
        ax.imshow(self.grid, cmap='binary', origin='lower', 
                 extent=[0, self.width, 0, self.height])
        
        # Highlight rooms with a different color
        for (x1, y1, x2, y2) in self.rooms:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                fill=False, edgecolor='blue', linewidth=2, label='Rooms')
            ax.add_patch(rect)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Corridor Environment ({len(self.rooms)} rooms, {self.num_corridors} corridors, {free_space:.1f}% free space)')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
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
        
        # Highlight rooms
        for (x1, y1, x2, y2) in self.rooms:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                fill=False, edgecolor='blue', linewidth=1.5, alpha=0.6)
            ax.add_patch(rect)
        
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
        
        # Title with path info
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