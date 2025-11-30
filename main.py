from project_root.environment.RandomEnvironment import RandomEnvironment

if __name__ == "__main__":
    # Create random environment
    env = RandomEnvironment(width=50, height=50, density=10, seed=42)
    print(f"   Environment created: {env.width}x{env.height}, density={env.density}%")
    # Visualize just the environment
    env.visualize()

    # Sample usage for planner
    from project_root.planner import RRTPlanner
    planner = RRTPlanner(env, start=(5, 5), goal=(45, 45))
    path = planner.search()
    planner.show_path() 