"""
gnn_model.py

Small graph neural network arm for train_models.py (plain PyTorch, no PyG).

With ~59 labelled updates a GNN is expected to overfit, so this is kept
deliberately tiny and is reported as an exploratory arm only:

  * Input graph per sample: the *impact subgraph* of the updated dependency,
    i.e. its DEPENDENCY node(s), the API nodes it provides, every FUNCTION that
    reaches those APIs within analyzer.MAX_BFS_DEPTH hops, and their FILE nodes.
    Samples whose dependency is not found in the DIG get a single "empty" node,
    so the network falls back to the SemVer features for them.
  * Node features: node-type one-hot (4), target-dependency/API flag, hop depth,
    log in-degree and log out-degree in the full DIG.
  * Model: 2 GCN layers (hidden 16, symmetric-normalized adjacency with
    self-loops) -> mean+max pooling -> concat standardized SemVer features ->
    2-layer MLP -> P(breaking).
  * Training: full-batch Adam, fixed 150 epochs (no early stopping, which would
    need a validation split we can't afford), weight decay + dropout,
    class-weighted BCE (pos_weight = n_safe / n_breaking in the training fold).

gnn_oof_probs() uses the exact same StratifiedGroupKFold splits as the other
arms so its out-of-fold probabilities are directly comparable.
"""

from __future__ import annotations

import collections
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer import find_matching_dependency_nodes, traverse_impacted_callsites

NODE_TYPES = ["FILE", "FUNCTION", "API", "DEPENDENCY"]
N_NODE_FEATS = len(NODE_TYPES) + 4
MAX_SUBGRAPH_NODES = 3000


def build_impact_subgraph(dig: Optional[dict], dependency: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (node_features [n, F], edge_index [2, m]) for the dependency's impact subgraph."""
    empty = (np.zeros((1, N_NODE_FEATS), dtype=np.float32), np.zeros((2, 0), dtype=np.int64))
    if dig is None:
        return empty

    nodes = dig.get("nodes", [])
    edges = dig.get("edges", [])
    matched = find_matching_dependency_nodes(nodes, dependency)
    if not matched:
        return empty
    func_depths, _apis, func_to_file = traverse_impacted_callsites(dig, matched)

    node_type = {n["id"]: n.get("type") for n in nodes}
    in_deg: collections.Counter = collections.Counter()
    out_deg: collections.Counter = collections.Counter()
    for e in edges:
        out_deg[e["source"]] += 1
        in_deg[e["target"]] += 1

    matched_set = set(matched)
    target_apis = {e["source"] for e in edges if e.get("type") == "PROVIDED_BY" and e["target"] in matched_set}

    # Keep the closest functions if the impact set is huge.
    funcs = sorted(func_depths, key=lambda f: func_depths[f])[:MAX_SUBGRAPH_NODES]
    files = {"file:" + func_to_file[f] for f in funcs if f in func_to_file}
    keep = list(matched) + sorted(target_apis) + funcs + sorted(files)
    keep = list(dict.fromkeys(keep))
    idx = {nid: i for i, nid in enumerate(keep)}

    x = np.zeros((len(keep), N_NODE_FEATS), dtype=np.float32)
    for nid, i in idx.items():
        t = node_type.get(nid)
        if t in NODE_TYPES:
            x[i, NODE_TYPES.index(t)] = 1.0
        x[i, 4] = 1.0 if (nid in matched_set or nid in target_apis) else 0.0
        if nid in func_depths:
            x[i, 5] = func_depths[nid] / 4.0
        x[i, 6] = math.log1p(in_deg[nid]) / 5.0
        x[i, 7] = math.log1p(out_deg[nid]) / 5.0

    src, dst = [], []
    for e in edges:
        s, t = idx.get(e["source"]), idx.get(e["target"])
        if s is not None and t is not None:
            src.append(s)
            dst.append(t)
    return x, np.array([src, dst], dtype=np.int64).reshape(2, -1)


def _norm_adj(n: int, edge_index: np.ndarray) -> torch.Tensor:
    """Symmetric-normalized sparse adjacency D^-1/2 (A + A^T + I) D^-1/2."""
    loops = np.arange(n)
    src = np.concatenate([edge_index[0], edge_index[1], loops])
    dst = np.concatenate([edge_index[1], edge_index[0], loops])
    pairs = np.unique(np.stack([src, dst]), axis=1)
    deg = np.bincount(pairs[0], minlength=n).astype(np.float32)
    w = 1.0 / np.sqrt(deg[pairs[0]] * deg[pairs[1]])
    return torch.sparse_coo_tensor(torch.from_numpy(pairs), torch.from_numpy(w), (n, n)).coalesce()


class TinyGCN(nn.Module):
    def __init__(self, n_tab: int, hidden: int = 16, dropout: float = 0.2, use_graph: bool = True):
        super().__init__()
        self.use_graph = use_graph
        self.g1 = nn.Linear(N_NODE_FEATS, hidden)
        self.g2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        head_in = (2 * hidden if use_graph else 0) + n_tab
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def embed(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.sparse.mm(adj, self.g1(x)))
        h = self.drop(h)
        h = torch.relu(torch.sparse.mm(adj, self.g2(h)))
        return torch.cat([h.mean(0), h.max(0).values])

    def forward(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]], tab: torch.Tensor) -> torch.Tensor:
        parts = [tab]
        if self.use_graph:
            parts.insert(0, torch.stack([self.embed(x, a) for x, a in graphs]))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def gnn_oof_probs(
    graphs_np: List[Tuple[np.ndarray, np.ndarray]],
    X_tab: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
    epochs: int = 150,
    use_graph: bool = True,
    n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold P(breaking) for the GNN, using the shared StratifiedGroupKFold protocol."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Single-threaded on purpose: the model is tiny, and on macOS torch and
    # xgboost each ship their own libomp, which deadlocks when both fork
    # OpenMP thread pools in the same process.
    torch.set_num_threads(1)
    graphs = [(torch.from_numpy(x), _norm_adj(len(x), ei)) for x, ei in graphs_np]

    oof = np.zeros(len(y), dtype=float)
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    for train_idx, val_idx in sgkf.split(X_tab, y, groups):
        mu = X_tab[train_idx].mean(0)
        sd = X_tab[train_idx].std(0)
        sd[sd == 0] = 1.0
        tab = torch.tensor((X_tab - mu) / sd, dtype=torch.float32)

        model = TinyGCN(n_tab=X_tab.shape[1], use_graph=use_graph)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        y_tr = torch.tensor(y[train_idx], dtype=torch.float32)
        n_pos = float(y_tr.sum())
        n_neg = float(len(y_tr) - n_pos)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1.0)))
        g_tr = [graphs[i] for i in train_idx]

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(g_tr, tab[train_idx]), y_tr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model([graphs[i] for i in val_idx], tab[val_idx])
        oof[val_idx] = torch.sigmoid(logits).numpy()
    return oof
