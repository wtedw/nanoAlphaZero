"""Portable adaptation of az/diagnostics/opening_value_charts.py board helpers.

Preserves AZ's table columns, palette, coordinates, highlights and square cells.
Plotly serializes numeric chart data; no raster images are generated or uploaded.
"""

import numpy as np


OPENING_VALUE_COLUMNS = [
    "game", "boardsize", "action", "row", "col", "value", "ground_truth", "highlight", "label",
]


def board_table(values, ground_truth=None, highlights=()):
    values = np.asarray(values)
    size = values.shape[0]
    highlights = set(map(int, highlights))
    rows = [["hex", size, r * size + c, r, c, float(values[r, c]),
             None if ground_truth is None else float(ground_truth[r, c]),
             r * size + c in highlights, f"r{r}c{c}"]
            for r in range(size) for c in range(size)]
    return dict(columns=OPENING_VALUE_COLUMNS, data=rows)


def board_figure(table, title, logits=False):
    import plotly.graph_objects as go
    size = table["data"][0][1]
    values = np.array([row[5] for row in table["data"]]).reshape(size, size)
    highlights = [row[2] for row in table["data"] if row[7]]
    label = "Logit" if logits else "Value"
    fig = go.Figure(go.Heatmap(
        z=values.tolist(), x=list(range(size)), y=list(range(size)),
        text=[[f"{x:+.2f}" for x in row] for row in values], texttemplate="%{text}",
        colorscale="Viridis" if logits else "RdBu", zmid=None if logits else 0.,
        colorbar=dict(title=label),
        hovertemplate=f"row=%{{y}}<br>col=%{{x}}<br>{label.lower()}=%{{z:.4f}}<extra></extra>"))
    if highlights:
        fig.add_trace(go.Scatter(
            x=[i % size for i in highlights], y=[i // size for i in highlights], mode="markers",
            marker=dict(symbol="square-open", size=34, color="red" if logits else "royalblue", line=dict(width=3)),
            name="Top-k" if logits else "Bottom-k", hoverinfo="skip"))
    fig.update_layout(title=title, xaxis=dict(title="Column", constrain="domain"),
                      yaxis=dict(title="Row", autorange="reversed", scaleanchor="x", scaleratio=1, constrain="domain"),
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def wandb_board_logs(tables):
    import wandb
    logs = {name: wandb.Table(**table) for name, table in tables.items()}
    for table_key, chart_key, title, logits in (
        ("model/values_after_black_moves_table", "model/values_after_black_moves_heatmap", "Value-Head After Each Possible Black Opening Move (Bottom-k in Blue)", False),
        ("model/logits_heatmap_top_k_table", "model/logits_heatmap_top_k", "Logits Heatmap (Top-k Highlighted)", True),
        ("muzero/model/latent_values_after_black_moves_table", "muzero/model/latent_values_after_black_moves_heatmap", "Learned-Latent Value After Each Black Opening (White perspective)", False),
    ):
        if table_key in tables:
            logs[chart_key] = wandb.Plotly(board_figure(tables[table_key], title, logits))
    return logs
