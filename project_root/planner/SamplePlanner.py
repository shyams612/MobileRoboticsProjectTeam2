class SamplePlanner:
    
    def __init__(self, start, goal, env):
        self.start = start
        self.goal = goal
        self.env = env
        self.path = [self.start]

    def search(self):
        """
         Return a list of waypoints from start to goal if a path is found, else empty list.
        """
        # use env.sample_free_point() to sample random points.
        count = 0
        while count <= 10:  # arbitrary number of waypoints
            point = self.env.sample_free_point()
            if self.env.is_collision_free(self.path[-1], point):
                self.path.append(point)
                count += 1
        
        while not self.env.is_collision_free(self.path[-1], self.goal):
            # keep sampling until we can connect to goal
            point = self.env.sample_free_point()
            if self.env.is_collision_free(self.path[-1], point):
                self.path.append(point)
        self.path.append(self.goal)
        return self.path

    def show_path(self):
        """
        Visualize the planning result by delegating to environment
        """
        self.env.show_path(
            start=self.start,
            goal=self.goal,
            path=self.path
        )