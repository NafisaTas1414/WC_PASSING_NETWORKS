"""
Build directed weighted passing networks from cleaned pass data.

Each network is keyed by (match_id, team).  Nodes are players; directed
edges carry the count of completed passes as their weight.  Node attributes
store the player's average pitch position derived from pass-start coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd


@dataclass
class PassNetwork:
    """Container for one team's passing network in one match."""
    match_id: int
    team: str
    graph: nx.DiGraph
    node_table: pd.DataFrame   # player, avg_x, avg_y, total_passes_made, total_passes_received
    edge_table: pd.DataFrame   # passer, recipient, weight


def build_network(
    passes: pd.DataFrame,
    match_id: int,
    team: str,
    min_passes: int = 1,
) -> PassNetwork:
    """
    Construct a directed weighted passing network for one team in one match.

    Parameters
    ----------
    passes    : tidy pass DataFrame (output of data_loader.extract_passes)
    match_id  : integer match identifier
    team      : team name string (must match 'team' column)
    min_passes: minimum edge weight to include in the graph
    """
    mask = (passes["match_id"] == match_id) & (passes["team"] == team)
    team_passes = passes[mask].dropna(subset=["player", "pass_recipient"])

    G = nx.DiGraph()

    # ---- edges ----------------------------------------------------------
    edge_counts: Dict[Tuple[str, str], int] = {}
    for _, row in team_passes.iterrows():
        src = row["player"]
        dst = row["pass_recipient"]
        edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1

    for (src, dst), weight in edge_counts.items():
        if weight >= min_passes:
            G.add_edge(src, dst, weight=weight)

    # ---- node positions (average pass-start x/y) -----------------------
    pos_data: Dict[str, List] = {}
    for _, row in team_passes.iterrows():
        player = row["player"]
        if player not in pos_data:
            pos_data[player] = {"x": [], "y": []}
        if pd.notna(row.get("x")):
            pos_data[player]["x"].append(row["x"])
        if pd.notna(row.get("y")):
            pos_data[player]["y"].append(row["y"])

    # Also note players who appear only as recipients
    for _, row in team_passes.iterrows():
        recipient = row["pass_recipient"]
        if recipient not in pos_data:
            pos_data[recipient] = {"x": [], "y": []}
        if pd.notna(row.get("end_x")):
            pos_data[recipient]["x"].append(row["end_x"])
        if pd.notna(row.get("end_y")):
            pos_data[recipient]["y"].append(row["end_y"])

    for player, coords in pos_data.items():
        avg_x = sum(coords["x"]) / len(coords["x"]) if coords["x"] else None
        avg_y = sum(coords["y"]) / len(coords["y"]) if coords["y"] else None
        if player in G.nodes:
            G.nodes[player]["avg_x"] = avg_x
            G.nodes[player]["avg_y"] = avg_y
        else:
            G.add_node(player, avg_x=avg_x, avg_y=avg_y)

    # ---- node table -----------------------------------------------------
    out_degree = dict(G.out_degree(weight="weight"))
    in_degree  = dict(G.in_degree(weight="weight"))
    node_rows = []
    for node in G.nodes:
        node_rows.append({
            "player": node,
            "avg_x": G.nodes[node].get("avg_x"),
            "avg_y": G.nodes[node].get("avg_y"),
            "passes_made":     out_degree.get(node, 0),
            "passes_received": in_degree.get(node, 0),
        })
    node_table = pd.DataFrame(node_rows)

    # ---- edge table -----------------------------------------------------
    edge_rows = [
        {"passer": u, "recipient": v, "weight": d["weight"]}
        for u, v, d in G.edges(data=True)
    ]
    edge_table = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
        columns=["passer", "recipient", "weight"]
    )

    return PassNetwork(
        match_id=match_id,
        team=team,
        graph=G,
        node_table=node_table,
        edge_table=edge_table,
    )


def build_all_networks(
    passes: pd.DataFrame,
    min_passes: int = 2,
) -> Dict[Tuple[int, str], PassNetwork]:
    """
    Build networks for every (match_id, team) combination in the pass table.

    Returns a dict keyed by (match_id, team).
    """
    networks: Dict[Tuple[int, str], PassNetwork] = {}
    groups = passes.dropna(subset=["player", "pass_recipient"]).groupby(
        ["match_id", "team"]
    )
    for (match_id, team), _ in groups:
        net = build_network(passes, int(match_id), team, min_passes=min_passes)
        networks[(int(match_id), team)] = net
    return networks


def save_network_tables(
    network: PassNetwork,
    output_dir,
) -> None:
    """Persist edge list and node table for a single network to CSV."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"match{network.match_id}_{network.team.replace(' ', '_')}"
    network.edge_table.to_csv(out / f"{stem}_edges.csv", index=False)
    network.node_table.to_csv(out / f"{stem}_nodes.csv", index=False)
