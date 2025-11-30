class RRTPlanner:
    
    def __init__(self, start, goal, env):
        self.start = start
        self.goal = goal
        self.env = env
        self.path = []

    def search(self):
        """
         Return a list of waypoints from start to goal if a path is found, else empty list.
        """
        # use env.sample_free_point() to sample random points.
        pass

    def show_path(self):
        """
        Visualize the planning result by delegating to environment
        """
        self.env.show_path(
            start=self.start,
            goal=self.goal,
            path=self.path
        )