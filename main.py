from project_root.environment.RandomEnvironment import RandomEnvironment
from project_root.planner.SamplePlanner import SamplePlanner
from project_root.planner.RRTStar import RRTStar
from project_root.planner.RRTStarDubins import RRTStarDubins
from project_root.environment.CorridorEnvironment import CorridorEnvironment
import math


if __name__ == "__main__":
    # Create random environment
    # env = RandomEnvironment(width=50, height=50, density=2, seed=42)
    env = CorridorEnvironment(width=50, height=50, corridor_width=12, num_corridors=5, seed=400)

    print("\n1. Creating corridor environment...")
    # print(f"   Environment created: {env.width}x{env.height}, density={env.density}%")
    # Visualize just the environment
    env.visualize()

    # # Sample usage for planner
    # print("*"*30)
    # print("Running RRT* planner...")
    # plannerA = RRTStar(start=(15, 5), goal=(45, 40), env=env)
    # path = plannerA.search()
    # plannerA.show_path() 

    # print("*"*30)
    # print("Running RRT* Dubins planner...")
    # plannerB = RRTStarDubins(start=(15, 5, math.pi/4), goal=(45, 40, 0), env=env)
    # path = plannerB.search()
    # plannerB.show_path() 


    # env = CorridorEnvironment(width=50, height=50, corridor_width=5, num_corridors=4, seed=42)
    # print("\n1. Creating corridor environment...")

    # Sample usage for planner
    print("*"*30)
    start = env.sample_free_point()
    goal = env.sample_free_point()
    print("Starting point:", start, "Goal point:", goal)
    print("Running RRT* planner...")
    plannerA = RRTStar(start=start, goal=goal, env=env)
    path = plannerA.search()
    plannerA.show_path() 

    print("*"*30)
    print("Running RRT* Dubins planner...")
    start1 = tuple(list(start) + [0])
    goal1 = tuple(list(goal) + [math.pi/2])
    plannerB = RRTStarDubins(start=start1, goal=goal1, env=env)
    path = plannerB.search()
    plannerB.show_path() 