import json
import time
import numpy as np
import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project_root.planner.RRTStar import RRTStar
from project_root.planner.RRTStarDubins import RRTStarDubins
from project_root.planner.BiRRTStar import BidirectionalRRTStar
from project_root.planner.BiRRTStarDubins import BidirectionalRRTStarDubins
from project_root.environment.SquareCorridorEnvironment import SquareCorridorEnvironment
from project_root.planner.PRRTStar import PRRTStar
from project_root.planner.PRRTStarDubins import PRRTStarDubins

from project_root.environment.RandomEnvironment import RandomEnvironment
from project_root.environment.CorridorEnvironment import CorridorEnvironment


class ExperimentRunner:
    """Manages loading and running path planning experiments from JSON config"""
    
    def __init__(self, config_path: str = "experiments.json", num_runs: int = 1):
        self.config_path = config_path
        self.num_runs = num_runs  # NEW: Number of times to run each experiment
        self.experiments = []
        self.results = []  # Now stores results from all runs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiment_runs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Map algorithm names to classes
        self.planner_classes = {
            "RRTStar": RRTStar,
            "RRTStarDubins": RRTStarDubins,
            "BiRRTStar": BidirectionalRRTStar,
            "BiRRTStarDubins": BidirectionalRRTStarDubins,
            "PRRTStar": PRRTStar,
            "PRRTStarDubins": PRRTStarDubins,
        }
        
        # Map environment names to classes
        self.env_classes = {
            "RandomEnvironment": RandomEnvironment,
            "CorridorEnvironment": CorridorEnvironment,
            "SquareCorridorEnvironment": SquareCorridorEnvironment
        }
    
    def load_experiments(self):
        """Load experiments from JSON configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.experiments = config.get("experiments", [])
            print(f"✓ Loaded {len(self.experiments)} experiments from {self.config_path}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: Config file '{self.config_path}' not found")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON in config file: {e}")
            return False
    
    def create_environment(self, env_config: Dict[str, Any]):
        """Create environment from configuration"""
        env_name = env_config.get("name")
        env_params = env_config.get("params", {})
        
        if env_name not in self.env_classes:
            raise ValueError(f"Unknown environment: {env_name}")
        
        env_class = self.env_classes[env_name]
        return env_class(**env_params)
    
    def parse_start_goal(self, experiment: Dict, env) -> Tuple:
        """Parse start and goal positions, handling 'sample' keyword"""
        start = experiment.get("start")
        goal = experiment.get("goal")
        
        # Handle sampling
        if start == "sample":
            start = env.sample_free_point()
        else:
            start = tuple(start)
        
        if goal == "sample":
            goal = env.sample_free_point()
        else:
            goal = tuple(goal)
        
        return start, goal
    
    def calculate_path_length(self, path: List[Tuple]) -> float:
        """Calculate total path length"""
        if path is None or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total_length += np.sqrt(dx**2 + dy**2)
        
        return total_length
    
    def generate_path_visualization(self, planner, algo_name: str) -> str:
        """Generate visualization and return as base64 encoded image"""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            env = planner.env
            
            # Draw environment
            ax.imshow(env.grid, cmap='binary', origin='lower',
                     extent=[0, env.width, 0, env.height])
            
            # Draw tree edges based on planner type
            if hasattr(planner, 'start_tree'):  # Bidirectional
                for (a, b) in planner.start_edges:
                    ax.plot([a[0], b[0]], [a[1], b[1]], 'b-', alpha=0.15, linewidth=0.5)
                for (a, b) in planner.goal_edges:
                    ax.plot([a[0], b[0]], [a[1], b[1]], 'g-', alpha=0.15, linewidth=0.5)
            else:  # Unidirectional
                for (a, b) in planner.all_edges:
                    if len(a) == 2:
                        ax.plot([a[0], b[0]], [a[1], b[1]], 'c-', alpha=0.15, linewidth=0.5)
                    else:
                        ax.plot([a[0], b[0]], [a[1], b[1]], 'c-', alpha=0.15, linewidth=0.5)
            
            # Draw final path
            if planner.final_path is not None:
                path = planner.final_path
                
                if len(path[0]) == 3 and 'Dubins' in algo_name:
                    import dubins
                    for i in range(len(path) - 1):
                        q0 = path[i]
                        q1 = path[i + 1]
                        dubins_path = dubins.shortest_path(q0, q1, planner.turning_radius)
                        configurations, _ = dubins_path.sample_many(0.2)
                        px = [c[0] for c in configurations]
                        py = [c[1] for c in configurations]
                        ax.plot(px, py, 'y-', linewidth=3, alpha=0.8)
                    
                    for i in range(0, len(path), max(1, len(path)//10)):
                        config = path[i]
                        ax.arrow(config[0], config[1],
                                0.8*np.cos(config[2]), 0.8*np.sin(config[2]),
                                head_width=0.5, head_length=0.5,
                                color='red', alpha=0.7)
                else:
                    px = [p[0] for p in path]
                    py = [p[1] for p in path]
                    ax.plot(px, py, 'y-', linewidth=3, label="Path", alpha=0.8)
                    ax.plot(px, py, 'yo', markersize=4)
            
            # Draw start and goal
            if hasattr(planner, 'start'):
                start = planner.start
                if hasattr(start, 'x'):
                    ax.plot(start.x, start.y, 'go', markersize=12, label='Start', 
                           markeredgecolor='darkgreen', markeredgewidth=2)
                    if hasattr(start, 'theta'):
                        ax.arrow(start.x, start.y,
                                1.5*np.cos(start.theta), 1.5*np.sin(start.theta),
                                head_width=0.5, head_length=0.5, fc='green', ec='green')
                else:
                    ax.plot(start[0], start[1], 'go', markersize=12, label='Start',
                           markeredgecolor='darkgreen', markeredgewidth=2)
            
            if hasattr(planner, 'goal'):
                goal = planner.goal
                if hasattr(goal, 'x'):
                    ax.plot(goal.x, goal.y, 'r*', markersize=16, label='Goal',
                           markeredgecolor='darkred', markeredgewidth=2)
                    if hasattr(goal, 'theta'):
                        ax.arrow(goal.x, goal.y,
                                1.5*np.cos(goal.theta), 1.5*np.sin(goal.theta),
                                head_width=0.5, head_length=0.5, fc='red', ec='red')
                else:
                    ax.plot(goal[0], goal[1], 'r*', markersize=16, label='Goal',
                           markeredgecolor='darkred', markeredgewidth=2)
            
            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_aspect('equal')
            ax.set_title(f"{algo_name} Path Planning")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return img_str
        except Exception as e:
            print(f"Warning: Could not generate visualization for {algo_name}: {e}")
            return ""
    
    def run_algorithm(self, algo_config: Dict, start: Tuple, goal: Tuple, env) -> Dict:
        """Run a single algorithm and return results"""
        algo_name = algo_config.get("name")
        algo_params = algo_config.get("params", {})
        
        if algo_name not in self.planner_classes:
            return {
                "success": False,
                "error": f"Unknown algorithm: {algo_name}"
            }
        
        original_start = start
        original_goal = goal
        if "Dubins" in algo_name:
            start_theta = algo_config.get("start_theta", 0.0)
            goal_theta = algo_config.get("goal_theta", 0.0)
            start = tuple(list(start) + [start_theta])
            goal = tuple(list(goal) + [goal_theta])
        
        planner_class = self.planner_classes[algo_name]
        planner = planner_class(start=start, goal=goal, env=env, **algo_params)
        
        print(f"  → Running {algo_name}...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            path = planner.search()
            elapsed_time = time.time() - start_time
            
            success = path is not None
            path_length = self.calculate_path_length(path) if success else 0.0
            num_nodes = len(planner.nodes) if hasattr(planner, 'nodes') else 0
            
            if hasattr(planner, 'start_tree') and hasattr(planner, 'goal_tree'):
                num_nodes = len(planner.start_tree) + len(planner.goal_tree)
            
            final_cost = None
            if hasattr(planner, 'best_cost') and planner.best_cost != float('inf'):
                final_cost = planner.best_cost
            elif hasattr(planner, 'goal_node_idx') and planner.goal_node_idx is not None:
                final_cost = planner.nodes[planner.goal_node_idx].cost
            
            visualization = ""
            if success:
                visualization = self.generate_path_visualization(planner, algo_name)
            
            print(f"✓ ({elapsed_time:.2f}s)")
            
            return {
                "success": success,
                "time": elapsed_time,
                "path_length": path_length,
                "num_nodes": num_nodes,
                "final_cost": final_cost,
                "path": path,
                "planner": planner,
                "visualization": visualization,
                "start": original_start,
                "goal": original_goal
            }
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"✗ Failed ({elapsed_time:.2f}s) - {str(e)}")
            return {
                "success": False,
                "time": elapsed_time,
                "error": str(e),
                "start": original_start,
                "goal": original_goal
            }
    
    def run_experiment(self, experiment: Dict, run_number: int):
        """Run a single experiment with all configured algorithms"""
        exp_name = experiment.get("name", "Unnamed")
        exp_desc = experiment.get("desc", "")
        
        print("\n" + "="*80)
        print(f"EXPERIMENT: {exp_name} [Run {run_number}/{self.num_runs}]")
        print(f"Description: {exp_desc}")
        print("="*80)
        
        # Create environment (with different seed for each run if RandomEnvironment)
        env_config = experiment["env"].copy()
        if env_config["name"] == "RandomEnvironment" and run_number > 1:
            # Modify seed for different runs to get different environments
            original_seed = env_config["params"].get("seed", 42)
            env_config["params"]["seed"] = original_seed + run_number - 1
            print(f"Using seed: {env_config['params']['seed']} for this run")
        
        env = self.create_environment(env_config)
        print(f"Environment: {env_config['name']} with params {env_config['params']}")
        
        # Get start and goal (will be different for each run if sampling)
        start, goal = self.parse_start_goal(experiment, env)
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        
        algorithms = experiment.get("algorithms", [])
        print(f"Found {len(algorithms)} algorithms")
        
        exp_results = {
            "run_number": run_number,
            "experiment_name": exp_name,
            "description": exp_desc,
            "environment": env_config,
            "start": start,
            "goal": goal,
            "algorithms": []
        }
        
        for idx, algo_config in enumerate(algorithms, 1):
            algo_name = algo_config.get("name")
            if not algo_config.get("enabled", True):
                print(f"[{idx}/{len(algorithms)}] Skipping {algo_name} (disabled)")
                continue
            print(f"[{idx}/{len(algorithms)}] Running {algo_name}...")
            result = self.run_algorithm(algo_config, start, goal, env)
            
            algo_result = {
                "name": algo_name,
                "params": algo_config.get("params"),
                **result
            }
            exp_results["algorithms"].append(algo_result)
        
        self.results.append(exp_results)
        self.print_experiment_summary(exp_results)
        
        # Save HTML report for this individual run
        self.save_html_report_for_run(exp_results)
    
    def print_experiment_summary(self, exp_results: Dict):
        """Print a summary table for an experiment"""
        print("\n" + "-"*80)
        print("RESULTS SUMMARY")
        print("-"*80)
        print(f"{'Algorithm':<35} {'Success':<10} {'Time (s)':<12} {'Path Len':<12} {'Nodes':<10} {'Cost':<10}")
        print("-"*80)
        
        for algo in exp_results["algorithms"]:
            name = algo["name"]
            if "early_stop" in algo.get("params", {}):
                early_stop_str = " [ES]" if algo["params"]["early_stop"] else " [FS]"
                name += early_stop_str
            
            success = "✓" if algo.get("success") else "✗"
            time_val = f"{algo.get('time', 0):.2f}"
            path_len = f"{algo.get('path_length', 0):.2f}" if algo.get("success") else "N/A"
            nodes = algo.get("num_nodes", 0)
            cost = f"{algo.get('final_cost', 0):.2f}" if algo.get("final_cost") is not None else "N/A"
            
            print(f"{name:<35} {success:<10} {time_val:<12} {path_len:<12} {nodes:<10} {cost:<10}")
        
        print("-"*80)
    
    def generate_html_report_for_run(self, exp_results: Dict) -> str:
        """Generate HTML report for a single run"""
        run_num = exp_results.get("run_number", 1)
        exp_name = exp_results.get("experiment_name", "Unnamed")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Path Planning Experiment - {exp_name} Run {run_num}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.3em;
            margin-bottom: 10px;
        }}
        
        .header .timestamp {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .summary {{
            background: #f8f9fa;
            padding: 30px 40px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .summary h2 {{
            color: #667eea;
            margin-bottom: 20px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .summary-card .label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .experiment {{
            padding: 40px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .experiment:last-child {{
            border-bottom: none;
        }}
        
        .experiment h2 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.8em;
        }}
        
        .experiment-desc {{
            color: #666;
            margin-bottom: 20px;
            font-style: italic;
        }}
        
        .experiment-meta {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 25px;
            font-family: monospace;
            font-size: 0.9em;
        }}
        
        .experiment-meta strong {{
            color: #667eea;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            background: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        thead {{
            background: #667eea;
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        .status-success {{
            color: #28a745;
            font-weight: bold;
        }}
        
        .status-failure {{
            color: #dc3545;
            font-weight: bold;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
            margin-left: 8px;
        }}
        
        .badge-es {{
            background: #ffc107;
            color: #000;
        }}
        
        .badge-fs {{
            background: #17a2b8;
            color: white;
        }}
        
        .visualizations {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }}
        
        .viz-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .viz-card .viz-header {{
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 2px solid #e9ecef;
            font-weight: 600;
            color: #333;
        }}
        
        .viz-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .viz-card .viz-footer {{
            padding: 15px;
            background: #f8f9fa;
            font-size: 0.9em;
            color: #666;
        }}
        
        .no-visualization {{
            padding: 40px;
            text-align: center;
            color: #999;
            font-style: italic;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media print {{
            body {{
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Path Planning Experiment</h1>
            <div class="subtitle">{exp_name} - Run {run_num}</div>
            <div class="timestamp">Run Date: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</div>
        </div>
        
        <div class="summary">
            <h2>📊 Run Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="label">Total Algorithms</div>
                    <div class="value">{len(exp_results['algorithms'])}</div>
                </div>
                <div class="summary-card">
                    <div class="label">Success Rate</div>
                    <div class="value">{self._calculate_run_success_rate(exp_results):.1f}%</div>
                </div>
                <div class="summary-card">
                    <div class="label">Total Time</div>
                    <div class="value">{self._calculate_run_total_time(exp_results):.1f}s</div>
                </div>
            </div>
        </div>
        
        <div class="experiment">
            <h2>Experiment Details</h2>
            <div class="experiment-desc">{exp_results['description']}</div>
            
            <div class="experiment-meta">
                <strong>Environment:</strong> {exp_results['environment']['name']}<br>
                <strong>Parameters:</strong> {json.dumps(exp_results['environment']['params'], indent=2)}<br>
                <strong>Start:</strong> {exp_results['start']}<br>
                <strong>Goal:</strong> {exp_results['goal']}
            </div>
            
            <h3>Algorithm Results</h3>
            <table>
                <thead>
                    <tr>
                        <th>Algorithm</th>
                        <th>Status</th>
                        <th>Time (s)</th>
                        <th>Path Length</th>
                        <th>Nodes</th>
                        <th>Cost</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for algo in exp_results['algorithms']:
            name = algo['name']
            if 'early_stop' in algo.get('params', {}):
                badge = '<span class="badge badge-es">ES</span>' if algo['params']['early_stop'] else '<span class="badge badge-fs">FS</span>'
                name += badge
            
            status_class = 'status-success' if algo.get('success') else 'status-failure'
            status_text = '✓ Success' if algo.get('success') else '✗ Failed'
            time_val = f"{algo.get('time', 0):.2f}"
            path_len = f"{algo.get('path_length', 0):.2f}" if algo.get('success') else "N/A"
            nodes = algo.get('num_nodes', 0)
            cost = f"{algo.get('final_cost', 0):.2f}" if algo.get('final_cost') is not None else "N/A"
            
            html += f"""
                    <tr>
                        <td>{name}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{time_val}</td>
                        <td>{path_len}</td>
                        <td>{nodes}</td>
                        <td>{cost}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
            
            <h3>Path Visualizations</h3>
            <div class="visualizations">
"""
        
        has_viz = False
        for algo in exp_results['algorithms']:
            if algo.get('success') and algo.get('visualization'):
                has_viz = True
                name = algo['name']
                if 'early_stop' in algo.get('params', {}):
                    suffix = ' [Early Stop]' if algo['params']['early_stop'] else ' [Full Search]'
                    name += suffix
                
                html += f"""
                <div class="viz-card">
                    <div class="viz-header">{name}</div>
                    <img src="data:image/png;base64,{algo['visualization']}" alt="{name} visualization">
                    <div class="viz-footer">
                        Path Length: {algo.get('path_length', 0):.2f} | 
                        Nodes: {algo.get('num_nodes', 0)} | 
                        Time: {algo.get('time', 0):.2f}s
                    </div>
                </div>
"""
        
        if not has_viz:
            html += '<div class="no-visualization">No successful paths to visualize</div>'
        
        html += """
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Path Planning Experiment Framework</p>
            <p>Configuration: """ + self.config_path + """ | Timestamp: """ + self.timestamp + """</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _calculate_run_success_rate(self, exp_results: Dict) -> float:
        """Calculate success rate for a single run"""
        total = len(exp_results['algorithms'])
        successes = sum(1 for algo in exp_results['algorithms'] if algo.get('success'))
        return (successes / total * 100) if total > 0 else 0.0
    
    def _calculate_run_total_time(self, exp_results: Dict) -> float:
        """Calculate total time for a single run"""
        return sum(algo.get('time', 0.0) for algo in exp_results['algorithms'])
    
    def save_html_report_for_run(self, exp_results: Dict) -> str:
        """Generate and save HTML report for a single run"""
        html_content = self.generate_html_report_for_run(exp_results)
        
        run_num = exp_results.get("run_number", 1)
        exp_name = exp_results.get("experiment_name", "Unnamed").replace(" ", "_")
        
        output_file = self.output_dir / f"run_{run_num}_{exp_name}_{self.timestamp}.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  ✓ HTML report saved to: {output_file}")
        
        return output_file
    
    def run_all_experiments(self):
        """Run all loaded experiments multiple times"""
        if not self.experiments:
            print("No experiments loaded. Call load_experiments() first.")
            return
        
        print(f"\n{'#'*80}")
        print(f"RUNNING {len(self.experiments)} EXPERIMENTS x {self.num_runs} RUNS EACH")
        print(f"TOTAL: {len(self.experiments) * self.num_runs} experiment runs")
        print(f"{'#'*80}")
        
        start_time = time.time()
        
        for i, experiment in enumerate(self.experiments, 1):
            if not experiment.get("enabled", True):
                print(f"\n[Experiment {i}/{len(self.experiments)}] SKIPPED (disabled)")
                continue
            
            print(f"\n{'='*80}")
            print(f"EXPERIMENT GROUP {i}/{len(self.experiments)}: {experiment.get('name')}")
            print(f"{'='*80}")
            
            # Run this experiment num_runs times
            for run_num in range(1, self.num_runs + 1):
                self.run_experiment(experiment, run_num)
        
        total_time = time.time() - start_time
        print(f"\n{'#'*80}")
        print(f"ALL EXPERIMENTS COMPLETED in {total_time:.2f}s")
        print(f"Total runs: {len(self.results)}")
        print(f"{'#'*80}")
    
    def save_json_results(self):
        """Save results to JSON file - now includes all runs"""
        results_to_save = []
        for exp in self.results:
            exp_copy = exp.copy()
            exp_copy["algorithms"] = []
            for algo in exp["algorithms"]:
                algo_copy = {k: v for k, v in algo.items() 
                           if k not in ["planner", "path", "visualization"]}
                exp_copy["algorithms"].append(algo_copy)
            results_to_save.append(exp_copy)
        
        output_file = self.output_dir / f"multi_run_{self.timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "num_runs": self.num_runs,
                    "timestamp": self.timestamp,
                    "config_file": self.config_path,
                    "total_experiment_runs": len(results_to_save)
                },
                "results": results_to_save
            }, f, indent=2)
        
        print(f"✓ JSON results saved to: {output_file}")
        return output_file


def main():
    """Main entry point for running experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run path planning experiments multiple times")
    parser.add_argument("--config", default="experiments.json", 
                       help="Path to experiments config file (default: experiments.json)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of times to run each experiment (default: 1)")
    parser.add_argument("--experiment", type=int,
                       help="Run only specific experiment number (1-indexed)")
    
    args = parser.parse_args()
    
    # Create and run experiments
    runner = ExperimentRunner(config_path=args.config, num_runs=args.runs)
    
    if not runner.load_experiments():
        return
    
    print(f"\nOutput directory: {runner.output_dir.absolute()}")
    print(f"Number of runs per experiment: {args.runs}")
    
    # Run specific experiment or all
    if args.experiment is not None:
        if 1 <= args.experiment <= len(runner.experiments):
            print(f"\nRunning only experiment {args.experiment}")
            for run_num in range(1, args.runs + 1):
                runner.run_experiment(runner.experiments[args.experiment - 1], run_num)
        else:
            print(f"Error: Experiment {args.experiment} not found. Valid range: 1-{len(runner.experiments)}")
            return
    else:
        runner.run_all_experiments()
    
    # Save results
    runner.save_json_results()


if __name__ == "__main__":
    main()