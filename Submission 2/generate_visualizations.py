#!/usr/bin/env python3
"""
Generate visualizations for Energy-Aware Peer Selection Report
Creates three plots: histogram, scatter plot, and sorted peer scores
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

print("Generating visualizations for Energy-Aware Peer Selection Report...")

# Generate synthetic data for 200 peers
n_peers = 200

# Generate upload speeds (5-20 MB/s) with some correlation to energy consumption
base_upload_speed = np.random.uniform(5, 20, n_peers)

# Add some noise and create outliers
noise = np.random.normal(0, 1.5, n_peers)
upload_speed = base_upload_speed + noise

# Create some outliers (5% of data)
outlier_indices = np.random.choice(n_peers, size=int(0.05 * n_peers), replace=False)
upload_speed[outlier_indices] = np.random.uniform(25, 30, len(outlier_indices))

# Ensure no negative values
upload_speed = np.clip(upload_speed, 1, None)

# Generate energy consumption (30-120 W) with inverse correlation to upload speed
base_energy = 150 - (upload_speed * 3) + np.random.normal(0, 15, n_peers)

# Add some outliers for energy consumption
energy_outlier_indices = np.random.choice(n_peers, size=int(0.03 * n_peers), replace=False)
base_energy[energy_outlier_indices] = np.random.uniform(140, 180, len(energy_outlier_indices))

# Ensure energy consumption is within reasonable bounds
energy_consumption = np.clip(base_energy, 20, 200)

# Calculate peer scores (upload_speed / energy_consumption)
peer_scores = upload_speed / energy_consumption

# Create DataFrame for easier manipulation
data = pd.DataFrame({
    'peer_id': range(1, n_peers + 1),
    'upload_speed': upload_speed,
    'energy_consumption': energy_consumption,
    'peer_score': peer_scores
})

print(f"Generated data for {n_peers} peers")
print(f"Upload speed range: {upload_speed.min():.2f} - {upload_speed.max():.2f} MB/s")
print(f"Energy consumption range: {energy_consumption.min():.2f} - {energy_consumption.max():.2f} W")

# Plot 1: Histogram of Upload Speeds
plt.figure(figsize=(10, 6))
plt.hist(upload_speed, bins=25, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
plt.title('Distribution of Upload Speeds\n(BitTorrent Peer Network)', fontsize=14, fontweight='bold')
plt.xlabel('Upload Speed (MB/s)', fontsize=12)
plt.ylabel('Number of Peers', fontsize=12)
plt.grid(True, alpha=0.3)

# Add statistics text
mean_speed = np.mean(upload_speed)
std_speed = np.std(upload_speed)
plt.axvline(mean_speed, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_speed:.2f} MB/s')
plt.legend()

# Add text box with statistics
textstr = f'n = {n_peers}\nMean = {mean_speed:.2f} MB/s\nStd = {std_speed:.2f} MB/s'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.75, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Save the histogram
plt.savefig('images/upload_speed_histogram.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print("✓ Saved: upload_speed_histogram.png")

# Plot 2: Scatter Plot - Energy Consumption vs Upload Speed
plt.figure(figsize=(10, 6))

# Create scatter plot with color mapping based on peer scores
scatter = plt.scatter(energy_consumption, upload_speed, 
                     c=peer_scores, cmap='viridis', 
                     alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

plt.title('Energy Consumption vs Upload Speed\n(Energy-Aware Peer Selection)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Energy Consumption (W)', fontsize=12)
plt.ylabel('Upload Speed (MB/s)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Peer Score (Speed/Energy)', fontsize=12)

# Add trend line
z = np.polyfit(energy_consumption, upload_speed, 1)
p = np.poly1d(z)
plt.plot(energy_consumption, p(energy_consumption), "r--", alpha=0.8, linewidth=2)

# Calculate correlation
correlation = np.corrcoef(energy_consumption, upload_speed)[0, 1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save the scatter plot
plt.savefig('images/energy_vs_upload_speed_scatter.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print("✓ Saved: energy_vs_upload_speed_scatter.png")

# Plot 3: Energy-Aware Peer Scores (Sorted)
# Sort data by peer scores for better visualization
data_sorted = data.sort_values('peer_score', ascending=False).reset_index(drop=True)

plt.figure(figsize=(14, 8))

# Create bar plot for top 50 peers (for readability)
top_n = 50
top_data = data_sorted.head(top_n)

bars = plt.bar(range(len(top_data)), top_data['peer_score'], 
               color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)

# Color code bars based on score ranges
for i, bar in enumerate(bars):
    score = top_data.iloc[i]['peer_score']
    if score > 0.15:
        bar.set_color('darkgreen')
    elif score > 0.12:
        bar.set_color('orange')
    else:
        bar.set_color('lightcoral')

plt.title(f'Top {top_n} Energy-Aware Peer Scores\n(Upload Speed / Energy Consumption)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Peer Rank', fontsize=12)
plt.ylabel('Energy-Aware Score', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars
for i, (bar, score) in enumerate(zip(bars, top_data['peer_score'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{score:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

# Add legend for color coding
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='darkgreen', label='High Efficiency (>0.15)'),
                   Patch(facecolor='orange', label='Medium Efficiency (0.12-0.15)'),
                   Patch(facecolor='lightcoral', label='Low Efficiency (<0.12)')]
plt.legend(handles=legend_elements, loc='upper right')

# Save the peer scores plot
plt.savefig('images/peer_scores_sorted.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print("✓ Saved: peer_scores_sorted.png")

# Summary Statistics
print("\n" + "="*60)
print("ENERGY-AWARE PEER SELECTION ANALYSIS SUMMARY")
print("="*60)

print(f"\nDataset Overview:")
print(f"• Total Peers: {n_peers}")
print(f"• Upload Speed Range: {upload_speed.min():.2f} - {upload_speed.max():.2f} MB/s")
print(f"• Energy Consumption Range: {energy_consumption.min():.2f} - {energy_consumption.max():.2f} W")
print(f"• Peer Score Range: {peer_scores.min():.4f} - {peer_scores.max():.4f}")

print(f"\nTop 10 Most Energy-Efficient Peers:")
print("-" * 50)
top_10 = data_sorted.head(10)[['peer_id', 'upload_speed', 'energy_consumption', 'peer_score']]
print(top_10.to_string(index=False, float_format='%.3f'))

print(f"\nBottom 10 Least Energy-Efficient Peers:")
print("-" * 50)
bottom_10 = data_sorted.tail(10)[['peer_id', 'upload_speed', 'energy_consumption', 'peer_score']]
print(bottom_10.to_string(index=False, float_format='%.3f'))

print(f"\nCorrelation Analysis:")
print(f"• Energy vs Upload Speed Correlation: {correlation:.3f}")
print(f"• Mean Peer Score: {peer_scores.mean():.4f}")
print(f"• Standard Deviation of Peer Score: {peer_scores.std():.4f}")

print("\n✓ All visualizations generated successfully!")
print("Images saved: images/upload_speed_histogram.png, images/energy_vs_upload_speed_scatter.png, images/peer_scores_sorted.png") 