# src/graph_analyzer.py
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for HPC
import matplotlib.pyplot as plt

def build_failure_graph():
    print("Building Failure Dependency Graph...")
    
    # 1. Create Directed Graph
    G = nx.DiGraph()
    
    # 2. Define Nodes (Subsystems)
    nodes = ["Power Delivery", "Voltage Reg", "Thermal Sensor", "GPU Core", "Memory Controller", "Driver"]
    G.add_nodes_from(nodes)
    
    # 3. Define Edges (Propagation Paths)
    edges = [
        ("Power Delivery", "Voltage Reg"),
        ("Voltage Reg", "Thermal Sensor"),
        ("Voltage Reg", "GPU Core"),
        ("Thermal Sensor", "GPU Core"),
        ("Memory Controller", "GPU Core"),
        ("Driver", "GPU Core")
    ]
    G.add_edges_from(edges)
    
    # 4. Calculate Centrality (Identify Critical Nodes)
    centrality = nx.degree_centrality(G)
    
    # 5. Save Results
    print("Top Critical Nodes:")
    sorted_cent = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_cent:
        print(f" - {node}: {score:.4f}")
        
    # Save graph visualization
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', arrows=True)
    plt.savefig("logs/failure_graph.png")
    print("Graph saved to logs/failure_graph.png")

if __name__ == "__main__":
    build_failure_graph()
