from project_root.environment.RandomEnvironment import RandomEnvironment
from project_root.planner.SamplePlanner import SamplePlanner
from project_root.planner.RRTStar import RRTStar
from project_root.planner.RRTStarDubins import RRTStarDubins
from project_root.planner.BiRRTStar import BidirectionalRRTStar
from project_root.planner.BiRRTStarDubins import BidirectionalRRTStarDubins
from project_root.planner.ImpBiRRTStar import ImprovedBidirectionalRRTStar
from project_root.planner.ImpBiRRTStarDubins import BiDirectionalRRTStarAPF
from project_root.environment.CorridorEnvironment import CorridorEnvironment
import math


if __name__ == "__main__":
    # Create random environment
    env = RandomEnvironment(width=100, height=100, density=2, seed=None, robot_radius=2.0)
    # env = RandomEnvironment(width=50, height=50, density=20, seed=42)
    # env = CorridorEnvironment(width=50, height=50, corridor_width=12, num_corridors=5, seed=400)
    # env.visualize()

    # Sample usage for planner
    start = env.sample_free_point()
    goal = env.sample_free_point()

    # start = (10, 10)
    # goal = (90, 90)

    # print("*"*30)
    # print("Starting point:", start, "Goal point:", goal)
    # print("Running RRT* planner...")
    # planner = RRTStar(start=start, goal=goal, env=env, early_stop=False)
    # path = planner.search()
    # planner.show_path() 

    # print("*"*30)
    # print("Running RRT* Dubins planner...")
    # start1 = tuple(list(start) + [0])
    # goal1 = tuple(list(goal) + [math.pi/2])
    # planner = RRTStarDubins(start=start1, goal=goal1, env=env, early_stop=False)
    # path = planner.search()
    # planner.show_path() 

    print("*"*30)
    print("Starting point:", start, "Goal point:", goal)
    print("Running Bidirectional RRT planner...")
    planner = BidirectionalRRTStar(start=start, goal=goal, env=env, early_stop=True)
    path = planner.search()
    planner.show_path() 

    # print("*"*30)
    # print("Running Bidirectional RRT Dubins planner...")
    # start1 = tuple(list(start) + [0])
    # goal1 = tuple(list(goal) + [math.pi/2])
    # planner = BidirectionalRRTStarDubins(start=start1, goal=goal1, env=env, early_stop=False)
    # path = planner.search()
    # planner.show_path() 

    print("*"*30)
    print("Starting point:", start, "Goal point:", goal)
    print("Running Improved Bidirectional RRT planner...")
    planner = ImprovedBidirectionalRRTStar(start=start, goal=goal, env=env)
    path = planner.search()
    planner.show_path() 

    # print("*"*30)
    # print("Starting point:", start, "Goal point:", goal)
    # print("Running Improved Bidirectional RRT planner 2...")
    # planner = BiDirectionalRRTStarAPF(start=start, goal=goal, env=env)
    # path = planner.search()
    # planner.show_path() 
