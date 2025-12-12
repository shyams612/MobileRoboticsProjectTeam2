import json
import numpy as np
from scipy import stats
from collections import defaultdict

def load_data(json_file):
    """Load JSON data from file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def analyze_algorithm_performance(data):
    """Analyze overall performance for each algorithm."""
    
    # Group results by algorithm
    algo_data = defaultdict(lambda: {
        'times': [], 
        'path_lengths': [], 
        'nodes': [], 
        'successes': [],
        'final_costs': []
    })
    
    for result in data['results']:
        for algo in result['algorithms']:
            name = algo['name']
            algo_data[name]['times'].append(algo['time'])
            algo_data[name]['nodes'].append(algo['num_nodes'])
            algo_data[name]['successes'].append(1 if algo['success'] else 0)
            
            if algo['success']:
                algo_data[name]['path_lengths'].append(algo['path_length'])
                if algo['final_cost']:
                    algo_data[name]['final_costs'].append(algo['final_cost'])
    
    # Calculate statistics
    stats_summary = {}
    for algo, metrics in algo_data.items():
        stats_summary[algo] = {
            'avg_time': np.mean(metrics['times']),
            'std_time': np.std(metrics['times']),
            'avg_path_length': np.mean(metrics['path_lengths']) if metrics['path_lengths'] else None,
            'std_path_length': np.std(metrics['path_lengths']) if metrics['path_lengths'] else None,
            'avg_nodes': np.mean(metrics['nodes']),
            'std_nodes': np.std(metrics['nodes']),
            'success_rate': np.mean(metrics['successes']) * 100,
            'total_runs': len(metrics['successes'])
        }
    
    return stats_summary, algo_data

def analyze_dubins_algorithms(stats_summary):
    """Analyze Dubins-constrained algorithms specifically."""
    
    dubins_algos = ['RRTStarDubins', 'BiRRTStarDubins', 'PRRTStarDubins']
    
    print("\n" + "="*80)
    print("DUBINS ALGORITHM PERFORMANCE ANALYSIS")
    print("="*80)
    
    for algo in dubins_algos:
        if algo in stats_summary:
            s = stats_summary[algo]
            print(f"\n{algo}:")
            print(f"  Average execution time: {s['avg_time']:.3f}s (±{s['std_time']:.3f}s)")
            print(f"  Average path length: {s['avg_path_length']:.1f} units (±{s['std_path_length']:.1f})")
            print(f"  Average nodes explored: {s['avg_nodes']:.0f} (±{s['std_nodes']:.0f})")
            print(f"  Success rate: {s['success_rate']:.0f}% ({int(s['success_rate']*s['total_runs']/100)}/{s['total_runs']})")

def analyze_environment_effects(data):
    """Analyze performance across different environments."""
    
    env_data = defaultdict(lambda: defaultdict(lambda: {
        'times': [], 
        'path_lengths': [], 
        'nodes': [], 
        'successes': []
    }))
    
    for result in data['results']:
        env_name = result['experiment_name']
        for algo in result['algorithms']:
            algo_name = algo['name']
            env_data[env_name][algo_name]['times'].append(algo['time'])
            env_data[env_name][algo_name]['nodes'].append(algo['num_nodes'])
            env_data[env_name][algo_name]['successes'].append(1 if algo['success'] else 0)
            if algo['success']:
                env_data[env_name][algo_name]['path_lengths'].append(algo['path_length'])
    
    print("\n" + "="*80)
    print("ENVIRONMENT-SPECIFIC EFFECTS")
    print("="*80)
    
    for env_name, algos in env_data.items():
        print(f"\n{env_name}:")
        for algo_name, metrics in algos.items():
            if metrics['times']:
                avg_time = np.mean(metrics['times'])
                success_rate = np.mean(metrics['successes']) * 100
                print(f"  {algo_name}: {avg_time:.3f}s, {success_rate:.0f}% success")
    
    return env_data

def analyze_holonomic_to_dubins_transition(stats_summary):
    """Analyze the transition from Euclidean to Dubins motion."""
    
    transitions = {
        'RRTStar': 'RRTStarDubins',
        'BiRRTStar': 'BiRRTStarDubins',
        'PRRTStar': 'PRRTStarDubins'
    }
    
    print("\n" + "="*80)
    print("HOLONOMIC TO DUBINS TRANSITION ANALYSIS")
    print("="*80)
    
    transition_stats = {}
    
    for euclidean, dubins in transitions.items():
        if euclidean in stats_summary and dubins in stats_summary:
            e_stats = stats_summary[euclidean]
            d_stats = stats_summary[dubins]
            
            time_change = ((d_stats['avg_time'] - e_stats['avg_time']) / e_stats['avg_time']) * 100
            node_change = ((d_stats['avg_nodes'] - e_stats['avg_nodes']) / e_stats['avg_nodes']) * 100
            success_change = d_stats['success_rate'] - e_stats['success_rate']
            
            transition_stats[euclidean] = {
                'time_change': time_change,
                'node_change': node_change,
                'success_change': success_change
            }
            
            print(f"\n{euclidean} → {dubins}:")
            print(f"  Time change: {time_change:+.0f}%")
            print(f"  Node change: {node_change:+.0f}%")
            print(f"  Success change: {success_change:+.0f}%")
    
    return transition_stats

def perform_statistical_tests(algo_data):
    """Perform paired t-tests between Euclidean and Dubins variants."""
    
    transitions = {
        'RRTStar': 'RRTStarDubins',
        'BiRRTStar': 'BiRRTStarDubins',
        'PRRTStar': 'PRRTStarDubins'
    }
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS (Paired t-tests)")
    print("="*80)
    
    for euclidean, dubins in transitions.items():
        if euclidean in algo_data and dubins in algo_data:
            print(f"\n{euclidean} vs {dubins}:")
            
            # Get paired data (same length)
            e_times = algo_data[euclidean]['times']
            d_times = algo_data[dubins]['times']
            min_len = min(len(e_times), len(d_times))
            
            # Time test
            t_stat, p_val = stats.ttest_rel(e_times[:min_len], d_times[:min_len])
            print(f"  Time: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
            
            # Nodes test
            e_nodes = algo_data[euclidean]['nodes']
            d_nodes = algo_data[dubins]['nodes']
            t_stat, p_val = stats.ttest_rel(e_nodes[:min_len], d_nodes[:min_len])
            print(f"  Nodes: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

def analyze_dense_environment_failures(data):
    """Analyze failures specifically in dense environments."""
    
    print("\n" + "="*80)
    print("DENSE ENVIRONMENT FAILURE ANALYSIS")
    print("="*80)
    
    dense_results = [r for r in data['results'] if 'High Density' in r['experiment_name']]
    
    failure_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    for result in dense_results:
        for algo in result['algorithms']:
            total_counts[algo['name']] += 1
            if not algo['success']:
                failure_counts[algo['name']] += 1
    
    print("\nFailure rates in high-density environments:")
    for algo in sorted(total_counts.keys()):
        failure_rate = (failure_counts[algo] / total_counts[algo]) * 100
        print(f"  {algo}: {failure_rate:.0f}% ({failure_counts[algo]}/{total_counts[algo]})")

def analyze_per_node_efficiency(stats_summary):
    """Analyze time per node efficiency."""
    
    print("\n" + "="*80)
    print("PER-NODE EFFICIENCY ANALYSIS")
    print("="*80)
    
    for algo, stats in sorted(stats_summary.items()):
        time_per_node = (stats['avg_time'] / stats['avg_nodes']) * 1000  # Convert to ms
        print(f"{algo}: {time_per_node:.2f} ms/node")

def generate_latex_table(transition_stats):
    """Generate LaTeX table code."""
    
    print("\n" + "="*80)
    print("LATEX TABLE CODE")
    print("="*80)
    
    print("""
\\begin{table}[htbp]
\\caption{Impact of Dubins Constraints on Algorithm Performance}
\\centering
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Algorithm Transition} & \\textbf{Time Change} & \\textbf{Node Change} & \\textbf{Success Change} \\\\
\\hline""")
    
    for algo, stats in transition_stats.items():
        print(f"{algo}$^*$ $\\rightarrow$ Dubins   & {stats['time_change']:+.0f}\\%  & {stats['node_change']:+.0f}\\%  & {stats['success_change']:+.0f}\\%  \\\\")
    
    print("""\\hline
\\end{tabular}}
\\label{tab:dubins-impact}
\\end{table}
""")

def main(json_file):
    """Main analysis pipeline."""
    
    print("Loading data...")
    data = load_data(json_file)
    
    print(f"Loaded {len(data['results'])} experiment runs")
    print(f"Total experiments: {data['metadata']['total_experiment_runs']}")
    
    # Overall algorithm performance
    stats_summary, algo_data = analyze_algorithm_performance(data)
    
    # Dubins-specific analysis
    analyze_dubins_algorithms(stats_summary)
    
    # Environment effects
    env_data = analyze_environment_effects(data)
    
    # Holonomic to Dubins transition
    transition_stats = analyze_holonomic_to_dubins_transition(stats_summary)
    
    # Statistical tests
    perform_statistical_tests(algo_data)
    
    # Dense environment failures
    analyze_dense_environment_failures(data)
    
    # Per-node efficiency
    analyze_per_node_efficiency(stats_summary)
    
    # Generate LaTeX table
    generate_latex_table(transition_stats)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Usage: python analyze_results.py
    import sys
    if len(sys.argv) != 2:
        print("Usage: python results_analyser.py <results.json>")
        sys.exit(1)

    json_file = sys.argv[1]
    main(json_file)