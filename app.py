from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from src.train_model import run_pipeline
from src.data_utils import normalize_events
from src.path_analyzer import suspicious_subgraph, find_paths

st.set_page_config(page_title="GraphSentinel | Lateral Movement Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.025em; }
.stApp { background: radial-gradient(circle at 8% 0%, #172554 0, #0b1220 32%, #080d17 80%); color: #e5edf8; }
section[data-testid="stSidebar"] { background: #0c1422; border-right: 1px solid #23324a; }
div[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(22,36,58,.95), rgba(14,23,39,.95)); border: 1px solid #263954; padding: 16px 18px; border-radius: 16px; }
div[data-testid="stMetricLabel"] { color: #9fb2cc; }
div[data-testid="stMetricValue"] { color: #f0f6ff; }
div.stButton > button { border-radius: 10px; border: 1px solid #3b82f6; font-weight: 600; }
div[data-testid="stDataFrame"] { border: 1px solid #263954; border-radius: 12px; }
section.main > div { padding-top: 1.6rem; }
.small-muted { color: #91a4be; font-size: 0.9rem; }
.hero { padding: 1.25rem 1.5rem; border: 1px solid #263954; border-radius: 18px; background: linear-gradient(115deg, rgba(30,58,138,.36), rgba(15,23,42,.75)); margin-bottom: 1rem; }
.pill { display:inline-block; color:#93c5fd; border:1px solid #315b91; border-radius:999px; padding:4px 10px; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "demo_network_events.csv"

@st.cache_data(show_spinner=False)
def load_demo():
    return pd.read_csv(DATA_PATH)

def graph_figure(graph, events, threshold, max_nodes=100):
    if graph.number_of_nodes() == 0:
        return go.Figure()
    sub_nodes = list(graph.nodes())[:max_nodes]
    sub = graph.subgraph(sub_nodes).copy()
    pos = nx.spring_layout(sub, seed=23, k=0.9)
    edge_x, edge_y = [], []
    suspicious_pairs = set(zip(events.loc[events["gnn_score"] >= threshold, "source_host"].astype(str),
                               events.loc[events["gnn_score"] >= threshold, "destination_host"].astype(str)))
    normal_x, normal_y, alert_x, alert_y = [], [], [], []
    for u, v in sub.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        target = (alert_x, alert_y) if (u, v) in suspicious_pairs else (normal_x, normal_y)
        target[0].extend([x0, x1, None]); target[1].extend([y0, y1, None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal_x, y=normal_y, mode="lines", line=dict(width=1, color="rgba(100,130,170,.35)"), hoverinfo="none", name="Other connections"))
    fig.add_trace(go.Scatter(x=alert_x, y=alert_y, mode="lines", line=dict(width=2, color="#fb7185"), hoverinfo="none", name="Flagged connections"))
    xs, ys, texts, colors, sizes = [], [], [], [], []
    flagged_hosts = set(events.loc[events["gnn_score"] >= threshold, "source_host"].astype(str)) | set(events.loc[events["gnn_score"] >= threshold, "destination_host"].astype(str))
    degree = dict(sub.degree())
    for node in sub.nodes():
        x, y = pos[node]
        xs.append(x); ys.append(y)
        texts.append(f"Host: {node}<br>Connections: {degree.get(node,0)}<br>In flagged path: {'Yes' if node in flagged_hosts else 'No'}")
        colors.append("#fb7185" if node in flagged_hosts else "#60a5fa")
        sizes.append(14 + min(degree.get(node,0), 12))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=list(sub.nodes()), textposition="top center",
        textfont=dict(color="#dbeafe", size=10), hovertext=texts, hoverinfo="text",
        marker=dict(size=sizes, color=colors, line=dict(width=1.4, color="#0b1220")), name="Hosts"))
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe", family="DM Sans"), margin=dict(l=10,r=10,t=20,b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h", y=1.05))
    return fig

def metrics_bar(metrics_a, metrics_b):
    rows = []
    for model_name, metrics in [("GraphSAGE-style GNN", metrics_a), ("Random Forest baseline", metrics_b)]:
        for key in ["precision", "recall", "f1", "pr_auc"]:
            rows.append({"Model": model_name, "Metric": key.replace("_", " ").upper(), "Score": metrics.get(key, 0.0)})
    return px.bar(pd.DataFrame(rows), x="Metric", y="Score", color="Model", barmode="group",
                  range_y=[0,1], text_auto=".2f", template="plotly_dark",
                  color_discrete_sequence=["#60a5fa", "#34d399"])

if "raw_data" not in st.session_state:
    st.session_state.raw_data = load_demo()
if "results" not in st.session_state:
    st.session_state.results = None

with st.sidebar:
    st.markdown("## 🛡️ GraphSentinel")
    st.caption("Network behavior analytics")
    st.markdown("---")
    st.markdown("### Data source")
    source_mode = st.radio("Choose input", ["Built-in demo", "Upload CSV"], label_visibility="collapsed")
    uploaded = None
    if source_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload host communication events", type=["csv"])
        if uploaded is not None:
            try:
                st.session_state.raw_data = pd.read_csv(uploaded)
                st.success(f"Loaded {len(st.session_state.raw_data):,} rows")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
    else:
        if st.button("Reset to demo dataset", use_container_width=True):
            st.session_state.raw_data = load_demo()
            st.session_state.results = None
    st.markdown("---")
    st.markdown("### Model settings")
    epochs = st.slider("Training epochs", min_value=20, max_value=200, value=100, step=20)
    threshold = st.slider("Alert threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
    st.caption("Higher threshold means fewer alerts.")
    train_clicked = st.button("⚡ Train model & analyze", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Offline CSV analysis • No traffic capture • No blocking")

if train_clicked:
    with st.spinner("Preprocessing events, building graph, and training the GNN..."):
        try:
            st.session_state.results = run_pipeline(st.session_state.raw_data, epochs=epochs, threshold=threshold)
        except Exception as exc:
            st.session_state.results = None
            st.error(f"Analysis failed: {exc}")

st.markdown('<div class="hero"><span class="pill">CYBER DEFENSE • GRAPH AI</span><h1 style="margin:.55rem 0 .25rem 0;">Lateral Movement Detection</h1><div class="small-muted">Map host-to-host communications, score suspicious edges, and investigate potential movement paths.</div></div>', unsafe_allow_html=True)

raw = st.session_state.raw_data
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events loaded", f"{len(raw):,}")
valid_hosts = 0
try:
    tmp = normalize_events(raw)
    valid_hosts = len(set(tmp["source_host"]) | set(tmp["destination_host"]))
except Exception:
    pass
c2.metric("Unique hosts", f"{valid_hosts:,}")
if st.session_state.results is not None:
    res = st.session_state.results
    ev = res["events"]
    flagged = ev[ev["gnn_score"] >= threshold]
    c3.metric("Flagged events", f"{len(flagged):,}", f"{len(flagged)/max(1,len(ev))*100:.1f}% of events")
    c4.metric("Suspicious hosts", f"{len(set(flagged['source_host']) | set(flagged['destination_host'])):,}")
else:
    c3.metric("Flagged events", "—", "Train model to analyze")
    c4.metric("Suspicious hosts", "—", "Awaiting analysis")

if st.session_state.results is None:
    st.info("Ready to analyze. Select a data source in the sidebar, then click **Train model & analyze**. The built-in synthetic dataset lets you run the full pipeline immediately.")
    st.markdown("### What this prototype does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 01 · Build a graph")
        st.write("Hosts become nodes and source-to-destination events become directed edges.")
    with col2:
        st.markdown("#### 02 · Learn relationships")
        st.write("A GraphSAGE-style model aggregates neighboring-host information and classifies events.")
    with col3:
        st.markdown("#### 03 · Investigate alerts")
        st.write("Inspect flagged edges, connected paths, and test-set metrics.")
    st.markdown("### Preview of the input data")
    st.dataframe(raw.head(12), use_container_width=True, hide_index=True)
    st.stop()

res = st.session_state.results
events = res["events"].copy()
# Apply live threshold changes without retraining.
events["prediction"] = (events["gnn_score"] >= threshold).astype(int)
events["prediction_label"] = events["prediction"].map({0:"Normal", 1:"Suspicious"})
flagged = events[events["gnn_score"] >= threshold].copy()

tabs = st.tabs(["🗺️ Network graph", "🚨 Alerts & paths", "📈 Model performance", "🧾 Event explorer"])
with tabs[0]:
    left, right = st.columns([1.65, 1])
    with left:
        st.markdown("### Host communication map")
        st.plotly_chart(graph_figure(res["graph"], events, threshold), use_container_width=True, config={"displayModeBar": True})
        st.caption("Blue nodes are not currently part of a flagged edge; rose nodes touch at least one flagged edge. Graph layout is illustrative.")
    with right:
        st.markdown("### Graph summary")
        st.metric("Graph nodes", f"{res['node_count']:,}")
        st.metric("Communication events", f"{res['edge_count']:,}")
        st.metric("Suspicious edges", f"{len(flagged):,}")
        st.metric("Training epochs", f"{len(res['history']):,}")
        st.markdown("#### Training loss")
        st.line_chart(res["history"].set_index("epoch")["loss"], height=190)
with tabs[1]:
    st.markdown("### Suspicious communication events")
    if flagged.empty:
        st.success("No events exceed the current alert threshold. Try lowering the threshold to inspect more candidates.")
    else:
        show_cols = ["timestamp","source_host","destination_host","event_type","protocol","gnn_score","label","prediction_label"]
        show_cols = [c for c in show_cols if c in flagged.columns]
        display = flagged[show_cols].sort_values("gnn_score", ascending=False).copy()
        display["gnn_score"] = display["gnn_score"].map(lambda x: round(float(x), 3))
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download flagged events CSV", flagged.to_csv(index=False).encode("utf-8"), "graphsentinel_flagged_events.csv", "text/csv")
    st.markdown("### Candidate paths")
    suspicious_graph = suspicious_subgraph(events, threshold)
    paths = find_paths(suspicious_graph)
    if paths:
        for i, path in enumerate(paths[:10], 1):
            st.markdown(f"**Path {i} · {len(path)-1} hops**")
            st.code("  →  ".join(path), language="text")
    elif suspicious_graph.number_of_edges():
        st.write("Flagged edges exist, but no multi-hop path was found. Individual suspicious edges are still available above.")
    else:
        st.write("No suspicious subgraph at the current threshold.")
with tabs[2]:
    st.markdown("### Test-set model evaluation")
    st.caption("Metrics are computed on the held-out event split. The synthetic demo is for pipeline demonstration only.")
    gnn_m, rf_m = res["gnn_metrics"], res["baseline_metrics"]
    a,b,c,d = st.columns(4)
    a.metric("GNN F1", f"{gnn_m['f1']:.3f}")
    b.metric("GNN precision", f"{gnn_m['precision']:.3f}")
    c.metric("GNN recall", f"{gnn_m['recall']:.3f}")
    d.metric("GNN PR-AUC", f"{gnn_m.get('pr_auc',0):.3f}")
    st.plotly_chart(metrics_bar(gnn_m, rf_m), use_container_width=True)
    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown("#### GNN confusion matrix")
        cm = gnn_m["confusion_matrix"]
        st.dataframe(pd.DataFrame(cm, index=["Actual normal","Actual suspicious"], columns=["Predicted normal","Predicted suspicious"]), use_container_width=True)
    with cm2:
        st.markdown("#### Random Forest baseline")
        st.write({k: round(v, 3) for k, v in rf_m.items() if k not in ["confusion_matrix"]})
    st.warning("Interpret scores only in the context of the dataset, label quality, split strategy, and false-positive cost. Do not report synthetic demo metrics as real-world performance.")
with tabs[3]:
    st.markdown("### Event explorer")
    q1, q2, q3 = st.columns([1,1,1])
    with q1:
        host_options = ["All hosts"] + sorted(set(events["source_host"].astype(str)) | set(events["destination_host"].astype(str)))
        selected_host = st.selectbox("Filter by host", host_options)
    with q2:
        pred_filter = st.selectbox("Prediction", ["All", "Suspicious", "Normal"])
    with q3:
        label_filter = st.selectbox("Ground truth", ["All", "Suspicious", "Normal"])
    view = events.copy()
    if selected_host != "All hosts":
        view = view[(view["source_host"].astype(str) == selected_host) | (view["destination_host"].astype(str) == selected_host)]
    if pred_filter != "All":
        view = view[view["prediction_label"] == pred_filter]
    if label_filter != "All":
        view = view[view["label"].map({0:"Normal",1:"Suspicious"}) == label_filter]
    st.caption(f"Showing {len(view):,} of {len(events):,} events")
    st.dataframe(view[["timestamp","source_host","destination_host","event_type","protocol","bytes_sent","bytes_received","failed_login","label","gnn_score","prediction_label"]], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download all predictions CSV", events.to_csv(index=False).encode("utf-8"), "graphsentinel_predictions.csv", "text/csv")

st.markdown("---")
st.markdown('<div class="small-muted">GraphSentinel prototype · Defensive research and education · Alerts require human investigation.</div>', unsafe_allow_html=True)
