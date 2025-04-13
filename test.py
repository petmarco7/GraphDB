from functools import reduce
from py2neo import Graph, Node, Relationship

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# Connect to the Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Clear existing data (use with caution in production)
graph.delete_all()


def define_nodes() -> list:
    """Define the list of nodes."""

    nodes = []

    nodes.append(Node("Person", name="Alice", score=10))
    nodes.append(Node("Person", name="Bob", score=7))
    nodes.append(Node("Person", name="Carol", score=5))
    nodes.append(Node("Person", name="Eva", score=2))

    nodes.append(Node("Subject", name="Math"))
    nodes.append(Node("Subject", name="Poetry"))

    nodes.append(Node("Company", name="Neo4j"))
    nodes.append(Node("Company", name="Google"))

    return nodes


def get_node_by_label(nodes: list, label: str, name: str) -> Node:
    """Get a node by its label and name."""

    for node in nodes:
        if label in node.labels and node["name"] == name:
            return node

    return None


def get_relations(nodes: list) -> list:
    """Define the list of relationships."""

    relations = []

    alice = get_node_by_label(nodes, "Person", "Alice")
    bob = get_node_by_label(nodes, "Person", "Bob")
    carol = get_node_by_label(nodes, "Person", "Carol")
    eva = get_node_by_label(nodes, "Person", "Eva")

    math = get_node_by_label(nodes, "Subject", "Math")
    poetry = get_node_by_label(nodes, "Subject", "Poetry")

    neo = get_node_by_label(nodes, "Company", "Neo4j")
    google = get_node_by_label(nodes, "Company", "Google")

    relations.append(Relationship(alice, "KNOWS", bob))
    relations.append(Relationship(bob, "KNOWS", carol))
    relations.append(Relationship(alice, "KNOWS", eva))

    relations.append(Relationship(alice, "LIKES", math))
    relations.append(Relationship(bob, "LIKES", math))
    relations.append(Relationship(carol, "LIKES", poetry))

    relations.append(Relationship(alice, "WORKS_FOR", neo))
    relations.append(Relationship(bob, "WORKS_FOR", neo))
    relations.append(Relationship(eva, "WORKS_FOR", google))

    return relations


def plot_graph(graph):
    """Plot the graph."""

    query = """
    MATCH (n)-[r]->(m)
    RETURN n.name AS source, 
        head(labels(n)) AS source_label,
        type(r) AS relationship, 
        m.name AS target,
        head(labels(m)) AS target_label
    """
    results = graph.run(query)

    G = nx.DiGraph()
    for record in results:
        G.add_node(record["source"], label=record["source_label"])
        G.add_node(record["target"], label=record["target_label"])
        G.add_edge(record["source"], record["target"], label=record["relationship"])

    # Define layout and color mapping
    node_degrees = dict(G.degree())
    pos = nx.spring_layout(G, k=4, seed=42)  # Seed for reproducibility
    node_labels = nx.get_node_attributes(G, "label")
    unique_labels = list(set(node_labels.values()))
    label_to_num = {label: num for num, label in enumerate(unique_labels)}
    node_numbers = [label_to_num[node_labels[node]] for node in G.nodes]
    cmap = plt.colormaps.get_cmap("coolwarm")
    node_colors = [cmap(num / len(unique_labels)) for num in node_numbers]

    # Initialize the plot with a larger figure size
    plt.figure(figsize=(10, 6))

    # Draw nodes with sizes proportional to their degree and assigned colors
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=[node_degrees[node] * 300 for node in G.nodes],
        node_color=node_colors,
        alpha=0.9,
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Draw edges with specified width and transparency
    nx.draw_networkx_edges(
        G, pos, arrowstyle="-|>", arrowsize=15, edge_color="black", width=1.5, alpha=0.7
    )
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=10, font_color="black"
    )

    # Create legend for node colors
    legend_patches = [
        mpatches.Patch(color=cmap(num / len(unique_labels)), label=label)
        for num, label in enumerate(unique_labels)
    ]

    plt.tight_layout()
    plt.legend(handles=legend_patches, loc="best")
    plt.show()


# Add nodes to the graph
nodes = define_nodes()
graph.create(reduce(lambda x, y: x | y, nodes))

# Create relationships
relations = get_relations(nodes)
graph.create(reduce(lambda x, y: x | y, relations))

# Query the graph: Find all people known by Alice
query = """
MATCH (a:Person)-[:KNOWS]->(friend:Person) 
WHERE a.name = 'Alice'
RETURN friend.name AS friend_name
"""
results = graph.run(query)

# Print the results
print("People known by Alice:")
for record in results:
    print(record["friend_name"])

# Query the graph: Find all people who like a subject
query = """
MATCH (a:Person)-[:LIKES]->(subject:Subject)
RETURN a.name AS person, subject.name AS subject
"""

results = graph.run(query)

# Print the results
print("\nPeople who like a subject:")
for record in results:
    print(f"{record['person']} likes {record['subject']}")

# Query the graph: Find all people who work for a company
query = """
MATCH (a:Person)-[:WORKS_FOR]->(company:Company)
RETURN a.name AS person, company.name AS company
"""

results = graph.run(query)

# Print the results
print("\nPeople who work for a company:")
for record in results:
    print(f"{record['person']} works for {record['company']}")

# Query the graph: Find the person with the highest score
query = """
MATCH (a:Person)
RETURN a.name AS person, a.score AS score
ORDER BY score DESC
LIMIT 1
"""

results = graph.run(query)

# Print the results
print("\nPerson with the highest score:")
for record in results:
    print(f"{record['person']} has the highest score of {record['score']}")

# Query the graph: Find the two people with the shortest score difference
query = """
MATCH (a:Person), (b:Person)
WHERE id(a) < id(b)
RETURN a.name AS person1, b.name AS person2, abs(a.score - b.score) AS score_diff
ORDER BY score_diff ASC
LIMIT 1
"""

results = graph.run(query)

# Print the results
print("\nTwo people with the shortest score difference:")
for record in results:
    print(f"{record['person1']} and {record['person2']} have a score difference of {record['score_diff']}")


# Query the graph: Find all the people who are connected with indefinite depth
query = """
MATCH (a:Person)-[:KNOWS*]-(b:Person)
WHERE id(a) < id(b)
RETURN a.name AS person1, b.name AS person2
"""

results = graph.run(query)

# Print the results
print("\nPeople who are connected:")
for record in results:
    print(f"{record['person1']} is connected by 'knows' {record['person2']}")


# Plot the graph
plot_graph(graph)
