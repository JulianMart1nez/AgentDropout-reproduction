"""Save and reload a learned AgentDropout graph (Team 8, Oct 2026, for Table 6 domain transferability).

The released code keeps the learned topology only in memory. What the evaluation phase reads is:
  - spatial_masks / temporal_masks: 0/1 masks after node and edge dropout (0 = edge removed),
  - spatial_logits / temporal_logits: edges still allowed by the masks are sampled as
    Bernoulli(sigmoid(logit)) when optimized_spatial / optimized_temporal is True,
  - skip_nodes: the agent index dropped in each round by node dropout,
  - optimized_spatial / optimized_temporal: whether eval samples edges or uses the masks as is.
These are saved after training and restored before evaluation on another benchmark, so the
target run evaluates exactly the graph the source run would have evaluated.
"""
import json

import torch


def _tensors(params):
    return [p.detach().clone() for p in params] if isinstance(params, torch.nn.ParameterList) else params.detach().clone()


def save_graph_state(graph, path, **meta):
    state = {
        "spatial_masks": _tensors(graph.spatial_masks),
        "temporal_masks": _tensors(graph.temporal_masks),
        "spatial_logits": _tensors(graph.spatial_logits),
        "temporal_logits": _tensors(graph.temporal_logits),
        "skip_nodes": list(graph.skip_nodes),
        "optimized_spatial": bool(graph.optimized_spatial),
        "optimized_temporal": bool(graph.optimized_temporal),
        "num_nodes": len(graph.nodes),
        "rounds": graph.rounds,
        "diff": graph.diff,
        "meta": meta,
    }
    torch.save(state, path)
    print(f"Saved learned graph to {path}: skip_nodes={state['skip_nodes']} "
          f"meta={json.dumps(meta, default=str)}", flush=True)


def _assign(params, saved, name):
    if isinstance(params, torch.nn.ParameterList):
        if len(params) != len(saved):
            raise ValueError(f"{name}: graph has {len(params)} rounds, saved graph has {len(saved)}")
        for p, s in zip(params, saved):
            if p.shape != s.shape:
                raise ValueError(f"{name}: shape {tuple(p.shape)} vs saved {tuple(s.shape)}")
            p.data.copy_(s)
    else:
        if params.shape != saved.shape:
            raise ValueError(f"{name}: shape {tuple(params.shape)} vs saved {tuple(saved.shape)}")
        params.data.copy_(saved)


def load_graph_state(graph, path):
    state = torch.load(path, weights_only=False)
    if state["num_nodes"] != len(graph.nodes) or state["rounds"] != graph.rounds or state["diff"] != graph.diff:
        raise ValueError(f"saved graph ({state['num_nodes']} agents, {state['rounds']} rounds, diff={state['diff']}) "
                         f"does not match this run ({len(graph.nodes)} agents, {graph.rounds} rounds, diff={graph.diff})")
    for name in ("spatial_masks", "temporal_masks", "spatial_logits", "temporal_logits"):
        _assign(getattr(graph, name), state[name], name)
    graph.skip_nodes = list(state["skip_nodes"])
    graph.optimized_spatial = state["optimized_spatial"]
    graph.optimized_temporal = state["optimized_temporal"]
    print(f"Loaded learned graph from {path}: skip_nodes={graph.skip_nodes} "
          f"meta={json.dumps(state.get('meta', {}), default=str)}", flush=True)
