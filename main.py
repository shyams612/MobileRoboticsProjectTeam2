from project_root.environment.RandomEnvironment import RandomEnvironment

if __name__ == "__main__":
    # Create random environment
    env = RandomEnvironment(width=50, height=50, density=10, seed=42)
    print(f"   Environment created: {env.width}x{env.height}, density={env.density}%")
    # Visualize just the environment
    env.visualize()