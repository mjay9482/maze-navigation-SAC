import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_metrics(metrics_file):
    with open(metrics_file, 'r') as f:
        return json.load(f)

def analyze_metrics(metrics):
    """Analyze training metrics and return a summary dictionary."""
    summary = {}
    
    # Convert lists to numpy arrays for easier analysis
    rewards = np.array(metrics['episode_rewards'])
    steps = np.array(metrics['episode_steps'])
    success_rate = np.array(metrics['success_rate'])
    
    # Overall performance
    summary['final_success_rate'] = success_rate[-1] * 100
    summary['best_success_rate'] = np.max(success_rate) * 100
    summary['final_avg_reward'] = np.mean(rewards[-10:])
    summary['best_avg_reward'] = np.max([np.mean(rewards[i:i+10]) for i in range(0, len(rewards)-9, 10)])
    
    # Learning speed
    summary['episodes_to_50_percent'] = np.where(success_rate >= 0.5)[0][0] if len(np.where(success_rate >= 0.5)[0]) > 0 else -1
    summary['episodes_to_75_percent'] = np.where(success_rate >= 0.75)[0][0] if len(np.where(success_rate >= 0.75)[0]) > 0 else -1
    
    # Efficiency metrics
    summary['avg_steps_per_episode'] = np.mean(steps)
    summary['successful_episodes'] = int(np.sum(rewards >= 1.0))
    summary['total_episodes'] = len(rewards)
    
    # Training stability
    summary['reward_std_last_10'] = np.std(rewards[-10:])
    summary['steps_std_last_10'] = np.std(steps[-10:])
    
    return summary

def generate_report(metrics_file):
    """Generate a comprehensive report from the metrics file."""
    metrics = load_metrics(metrics_file)
    summary = analyze_metrics(metrics)
    
    report = []
    report.append("=== Training Performance Summary ===")
    report.append(f"\nOverall Performance:")
    report.append(f"- Final Success Rate: {summary['final_success_rate']:.1f}%")
    report.append(f"- Best Success Rate: {summary['best_success_rate']:.1f}%")
    report.append(f"- Final Average Reward (last 10 episodes): {summary['final_avg_reward']:.2f}")
    report.append(f"- Best Average Reward (10-episode window): {summary['best_avg_reward']:.2f}")
    
    report.append(f"\nLearning Speed:")
    if summary['episodes_to_50_percent'] != -1:
        report.append(f"- Episodes to 50% Success Rate: {summary['episodes_to_50_percent']}")
    if summary['episodes_to_75_percent'] != -1:
        report.append(f"- Episodes to 75% Success Rate: {summary['episodes_to_75_percent']}")
    
    report.append(f"\nEfficiency Metrics:")
    report.append(f"- Average Steps per Episode: {summary['avg_steps_per_episode']:.1f}")
    report.append(f"- Successful Episodes: {summary['successful_episodes']} out of {summary['total_episodes']}")
    
    report.append(f"\nTraining Stability:")
    report.append(f"- Reward Standard Deviation (last 10): {summary['reward_std_last_10']:.2f}")
    report.append(f"- Steps Standard Deviation (last 10): {summary['steps_std_last_10']:.2f}")
    
    return "\n".join(report)

def plot_metrics(metrics, save_dir='plots'):
    """Generate and save plots of key metrics."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Convert lists to numpy arrays
    rewards = np.array(metrics['episode_rewards'])
    success_rate = np.array(metrics['success_rate'])
    steps = np.array(metrics['episode_steps'])
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Success Rate
    plt.subplot(2, 2, 1)
    plt.plot(success_rate * 100)
    plt.title('Success Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    
    # Plot 2: Episode Rewards
    plt.subplot(2, 2, 2)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot 3: Steps per Episode
    plt.subplot(2, 2, 3)
    plt.plot(steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot 4: Moving Average Reward
    plt.subplot(2, 2, 4)
    window = 10
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), moving_avg)
    plt.title(f'{window}-Episode Moving Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

if __name__ == "__main__":
    # Find the most recent metrics file
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        print("No logs directory found!")
        exit(1)
        
    metrics_files = [f for f in os.listdir(log_dir) if f.startswith('metrics_') and f.endswith('_final.json')]
    if not metrics_files:
        print("No metrics files found!")
        exit(1)
        
    latest_file = max(metrics_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
    metrics_file = os.path.join(log_dir, latest_file)
    
    # Generate report
    report = generate_report(metrics_file)
    print(report)
    
    # Save report
    with open(os.path.join(log_dir, 'performance_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate plots
    metrics = load_metrics(metrics_file)
    plot_metrics(metrics)
    
    print("\nReport saved to 'logs/performance_report.txt'")
    print("Plots saved to 'plots/training_metrics.png'") 