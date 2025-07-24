#!/usr/bin/env python3
"""
SUMO and CloudReports Analysis for Energy-Aware Peer Selection
This script simulates the analysis capabilities of SUMO and CloudReports
for the BitTorrent peer network research.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import networkx as nx
from datetime import datetime, timedelta
import json

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("SUMO & CLOUDREPORTS ANALYSIS FOR ENERGY-AWARE PEER SELECTION")
print("="*60)

# Generate comprehensive dataset for analysis
n_peers = 200
n_datacenters = 2
n_hosts_per_dc = 5
n_vms_per_host = 4

print(f"\n📊 Generating comprehensive dataset...")
print(f"• Total Peers: {n_peers}")
print(f"• Datacenters: {n_datacenters}")
print(f"• Hosts per DC: {n_hosts_per_dc}")
print(f"• VMs per Host: {n_vms_per_host}")

# Generate peer network data (SUMO-like analysis)
peer_ids = range(1, n_peers + 1)
upload_speeds = np.random.uniform(5, 20, n_peers) + np.random.normal(0, 2, n_peers)
upload_speeds = np.clip(upload_speeds, 1, 30)

energy_consumption = 150 - (upload_speeds * 3) + np.random.normal(0, 15, n_peers)
energy_consumption = np.clip(energy_consumption, 20, 200)

# Add some outliers
outlier_indices = np.random.choice(n_peers, size=int(0.05 * n_peers), replace=False)
upload_speeds[outlier_indices] = np.random.uniform(25, 35, len(outlier_indices))
energy_consumption[outlier_indices] = np.random.uniform(180, 250, len(outlier_indices))

# Calculate peer scores
peer_scores = upload_speeds / energy_consumption

# Generate network topology (SUMO-like network analysis)
G = nx.random_geometric_graph(n_peers, 0.15, seed=42)
# Add some edges based on peer scores (better peers connect more)
for i in range(n_peers):
    for j in range(i+1, n_peers):
        if np.random.random() < 0.1:  # Base connection probability
            if peer_scores[i] > 0.1 and peer_scores[j] > 0.1:  # High-scoring peers connect more
                G.add_edge(i, j)

# Generate datacenter data (CloudReports-like analysis)
datacenter_data = []
for dc_id in range(n_datacenters):
    for host_id in range(n_hosts_per_dc):
        host_energy = np.random.uniform(80, 120)  # kWh
        host_latency = np.random.uniform(100, 200)  # ms
        host_cpu_util = np.random.uniform(0.3, 0.9)
        
        for vm_id in range(n_vms_per_host):
            vm_energy = host_energy * np.random.uniform(0.1, 0.3)
            vm_latency = host_latency + np.random.uniform(-20, 20)
            vm_cpu_util = host_cpu_util * np.random.uniform(0.5, 1.2)
            
            datacenter_data.append({
                'dc_id': dc_id + 1,
                'host_id': host_id + 1,
                'vm_id': vm_id + 1,
                'host_energy_kwh': host_energy,
                'host_latency_ms': host_latency,
                'host_cpu_util': host_cpu_util,
                'vm_energy_kwh': vm_energy,
                'vm_latency_ms': vm_latency,
                'vm_cpu_util': vm_cpu_util
            })

# Create DataFrames
peer_df = pd.DataFrame({
    'peer_id': peer_ids,
    'upload_speed_mbps': upload_speeds,
    'energy_consumption_w': energy_consumption,
    'peer_score': peer_scores,
    'neighbor_count': [G.degree(i) for i in range(n_peers)],
    'clustering_coeff': [nx.clustering(G, i) for i in range(n_peers)]
})

dc_df = pd.DataFrame(datacenter_data)

print("✓ Dataset generation complete!")

# ============================================================================
# SUMO-LIKE ANALYSIS: Network Topology and Traffic Flow
# ============================================================================

print(f"\n🚦 SUMO-LIKE ANALYSIS: Network Topology and Traffic Flow")
print("-" * 50)

# Network topology analysis
avg_degree = np.mean([G.degree(i) for i in range(n_peers)])
avg_clustering = nx.average_clustering(G)
network_density = nx.density(G)
connected_components = nx.number_connected_components(G)

print(f"Network Topology Metrics:")
print(f"• Average Node Degree: {avg_degree:.2f}")
print(f"• Average Clustering Coefficient: {avg_clustering:.3f}")
print(f"• Network Density: {network_density:.3f}")
print(f"• Connected Components: {connected_components}")

# Traffic flow analysis (simulating SUMO traffic simulation)
print(f"\nTraffic Flow Analysis:")
print(f"• Total Network Connections: {G.number_of_edges()}")
print(f"• Average Path Length: {nx.average_shortest_path_length(G):.2f}")
print(f"• Network Diameter: {nx.diameter(G)}")

# Create SUMO-like network visualization
plt.figure(figsize=(15, 10))

# Plot 1: Network Topology
plt.subplot(2, 3, 1)
pos = nx.spring_layout(G, seed=42)
node_colors = [peer_scores[i] for i in range(n_peers)]
scatter = nx.draw(G, pos, node_color=node_colors, cmap='viridis', 
        node_size=50, alpha=0.8, edge_color='gray')
plt.title('SUMO: Peer Network Topology\n(Color = Energy-Aware Score)', fontsize=12, fontweight='bold')
sm = plt.cm.ScalarMappable(cmap='viridis')
sm.set_array(node_colors)
plt.colorbar(sm, ax=plt.gca(), label='Peer Score')

# Plot 2: Traffic Flow Distribution
plt.subplot(2, 3, 2)
degree_sequence = [G.degree(i) for i in range(n_peers)]
plt.hist(degree_sequence, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
plt.title('SUMO: Traffic Flow Distribution\n(Node Degrees)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Connections')
plt.ylabel('Number of Peers')
plt.grid(True, alpha=0.3)

# Plot 3: Energy vs Network Position
plt.subplot(2, 3, 3)
centrality = nx.betweenness_centrality(G)
centrality_values = list(centrality.values())
plt.scatter(centrality_values, energy_consumption, c=peer_scores, cmap='viridis', alpha=0.7)
plt.title('SUMO: Energy vs Network Centrality', fontsize=12, fontweight='bold')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Energy Consumption (W)')
plt.colorbar(label='Peer Score')

# ============================================================================
# CLOUDREPORTS-LIKE ANALYSIS: Datacenter Performance and Energy
# ============================================================================

print(f"\n☁️ CLOUDREPORTS-LIKE ANALYSIS: Datacenter Performance and Energy")
print("-" * 50)

# Policy A: Energy-centric policy
policy_a_energy = dc_df['vm_energy_kwh'].mean() * 0.8  # 20% energy reduction
policy_a_latency = dc_df['vm_latency_ms'].mean() * 1.15  # 15% latency increase

# Policy B: Latency-centric policy  
policy_b_energy = dc_df['vm_energy_kwh'].mean() * 1.2  # 20% energy increase
policy_b_latency = dc_df['vm_latency_ms'].mean() * 0.85  # 15% latency reduction

print(f"Policy Comparison (CloudReports Analysis):")
print(f"Policy A (Energy-Centric):")
print(f"  • Average Energy: {policy_a_energy:.2f} kWh")
print(f"  • Average Latency: {policy_a_latency:.2f} ms")
print(f"  • Energy Efficiency: High")
print(f"  • Performance: Moderate")

print(f"\nPolicy B (Latency-Centric):")
print(f"  • Average Energy: {policy_b_energy:.2f} kWh")
print(f"  • Average Latency: {policy_b_latency:.2f} ms")
print(f"  • Energy Efficiency: Low")
print(f"  • Performance: High")

# Create CloudReports-like visualizations
# Plot 4: Datacenter Energy Consumption
plt.subplot(2, 3, 4)
dc_energy = dc_df.groupby('dc_id')['vm_energy_kwh'].sum()
bars = plt.bar(dc_energy.index, dc_energy.values, color=['lightcoral', 'lightblue'], alpha=0.8)
plt.title('CloudReports: Datacenter Energy Consumption', fontsize=12, fontweight='bold')
plt.xlabel('Datacenter ID')
plt.ylabel('Total Energy (kWh)')
plt.xticks(dc_energy.index)
for bar, value in zip(bars, dc_energy.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{value:.1f}', ha='center', va='bottom')

# Plot 5: Policy Comparison
plt.subplot(2, 3, 5)
policies = ['Policy A\n(Energy-Centric)', 'Policy B\n(Latency-Centric)']
energy_values = [policy_a_energy, policy_b_energy]
latency_values = [policy_a_latency, policy_b_latency]

x = np.arange(len(policies))
width = 0.35

bars1 = plt.bar(x - width/2, energy_values, width, label='Energy (kWh)', color='orange', alpha=0.7)
bars2 = plt.bar(x + width/2, latency_values, width, label='Latency (ms)', color='purple', alpha=0.7)

plt.title('CloudReports: Policy Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Policy Type')
plt.ylabel('Value')
plt.xticks(x, policies)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: VM Performance Distribution
plt.subplot(2, 3, 6)
plt.scatter(dc_df['vm_cpu_util'], dc_df['vm_latency_ms'], 
           c=dc_df['vm_energy_kwh'], cmap='viridis', alpha=0.7, s=50)
plt.title('CloudReports: VM Performance vs Energy', fontsize=12, fontweight='bold')
plt.xlabel('CPU Utilization')
plt.ylabel('Latency (ms)')
plt.colorbar(label='Energy (kWh)')

plt.tight_layout()
plt.savefig('sumo_cloudreports_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sumo_cloudreports_analysis.png")

# ============================================================================
# INTEGRATED ANALYSIS REPORT
# ============================================================================

print(f"\n📋 INTEGRATED ANALYSIS REPORT")
print("="*60)

# Peer selection recommendations
top_peers = peer_df.nlargest(10, 'peer_score')
bottom_peers = peer_df.nsmallest(10, 'peer_score')

print(f"\n🎯 ENERGY-AWARE PEER SELECTION RECOMMENDATIONS:")
print("-" * 50)
print(f"Top 5 Recommended Peers (High Efficiency):")
for _, peer in top_peers.head().iterrows():
    print(f"  Peer {int(peer['peer_id']):3d}: Score={peer['peer_score']:.3f}, "
          f"Speed={peer['upload_speed_mbps']:.1f} MB/s, "
          f"Energy={peer['energy_consumption_w']:.1f} W")

print(f"\n❌ Peers to Avoid (Low Efficiency):")
for _, peer in bottom_peers.head().iterrows():
    print(f"  Peer {int(peer['peer_id']):3d}: Score={peer['peer_score']:.3f}, "
          f"Speed={peer['upload_speed_mbps']:.1f} MB/s, "
          f"Energy={peer['energy_consumption_w']:.1f} W")

# Network efficiency metrics
print(f"\n📊 NETWORK EFFICIENCY METRICS:")
print("-" * 30)
print(f"• Average Peer Score: {peer_df['peer_score'].mean():.4f}")
print(f"• Score Standard Deviation: {peer_df['peer_score'].std():.4f}")
print(f"• Network Connectivity: {G.number_of_edges()}/{G.number_of_nodes()} = {network_density:.3f}")
print(f"• Energy-Performance Correlation: {np.corrcoef(upload_speeds, energy_consumption)[0,1]:.3f}")

# Datacenter optimization
print(f"\n☁️ DATACENTER OPTIMIZATION INSIGHTS:")
print("-" * 35)
print(f"• Policy A (Energy-Centric) saves {(policy_b_energy - policy_a_energy):.1f} kWh per VM")
print(f"• Policy B (Latency-Centric) reduces latency by {(policy_a_latency - policy_b_latency):.1f} ms")
print(f"• Recommended: Hybrid approach based on workload requirements")

# Generate JSON report for further analysis
analysis_report = {
    "timestamp": datetime.now().isoformat(),
    "peer_network": {
        "total_peers": n_peers,
        "network_density": network_density,
        "avg_clustering": avg_clustering,
        "connected_components": connected_components,
        "avg_degree": avg_degree
    },
    "energy_analysis": {
        "avg_peer_score": float(peer_df['peer_score'].mean()),
        "score_std": float(peer_df['peer_score'].std()),
        "energy_correlation": float(np.corrcoef(upload_speeds, energy_consumption)[0,1]),
        "top_peers": top_peers[['peer_id', 'peer_score', 'upload_speed_mbps', 'energy_consumption_w']].to_dict('records')
    },
    "datacenter_policies": {
        "policy_a": {
            "energy_kwh": float(policy_a_energy),
            "latency_ms": float(policy_a_latency),
            "efficiency": "high"
        },
        "policy_b": {
            "energy_kwh": float(policy_b_energy),
            "latency_ms": float(policy_b_latency),
            "efficiency": "low"
        }
    },
    "recommendations": {
        "peer_selection": "Prioritize peers with scores > 0.12",
        "network_optimization": f"Maintain connectivity density around {network_density:.3f}",
        "datacenter_policy": "Use Policy A for energy-sensitive workloads, Policy B for latency-sensitive workloads"
    }
}

with open('analysis_report.json', 'w') as f:
    json.dump(analysis_report, f, indent=2)

print(f"\n💾 Analysis report saved to: analysis_report.json")
print(f"📊 Visualization saved to: sumo_cloudreports_analysis.png")

print(f"\n✅ SUMO & CloudReports Analysis Complete!")
print("="*60) 