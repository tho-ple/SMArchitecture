import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import TSNE
import plotly.express as px
import os

# --------------------------------------------------------
# Page configuration
# --------------------------------------------------------
st.set_page_config(page_title="Analyze Requirements", page_icon="🧠", layout="wide")
st.title("🧠 Analyze & Cluster Requirements")

st.markdown("""
This page analyzes the previously entered requirements and groups them into **functional clusters**.
Each cluster often represents a potential **component or subsystem** in your system architecture.
Adjust clustering sensitivity below to explore how requirements group together.
""")

# --------------------------------------------------------
# Load requirements from session state or file
# --------------------------------------------------------
if "requirements" not in st.session_state or len(st.session_state["requirements"]) == 0:
    st.warning("⚠️ No requirements found. Please add or import them first on the **Manage Requirements** page.")
    st.stop()

reqs = st.session_state["requirements"]
texts = [r["text"] for r in reqs]

# --------------------------------------------------------
# Initialize vector DB (Chroma)
# --------------------------------------------------------
chroma_path = os.path.join(os.getcwd(), "chroma_store")
os.makedirs(chroma_path, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=chroma_path)
collection_name = "requirements"

# Get or create collection
try:
    collection = chroma_client.get_collection(collection_name)
except Exception:
    collection = chroma_client.create_collection(collection_name)

# Clear old entries safely
existing = collection.count()
if existing > 0:
    all_ids = [r["id"] for r in reqs]
    try:
        collection.delete(ids=all_ids)
    except Exception:
        pass

# --------------------------------------------------------
# Load embedding model
# --------------------------------------------------------
with st.spinner("Loading embedding model (this may take a few seconds)..."):
    model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------------
# Generate embeddings
# --------------------------------------------------------
with st.spinner("Generating embeddings for requirements..."):
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings)

if len(embeddings) == 0:
    st.error("No embeddings were generated — please check your requirements.")
    st.stop()

# Add to Chroma
try:
    collection.add(
        ids=[r["id"] for r in reqs],
        embeddings=embeddings.tolist(),
        documents=texts,
    )
except ValueError:
    st.error("Failed to add embeddings to vector DB. Try reloading the page.")
    st.stop()

st.success(f"✅ Added {len(reqs)} requirements to vector database.")

# --------------------------------------------------------
# Clustering with adaptive sensitivity
# --------------------------------------------------------
st.subheader("🔍 Clustering Settings")

dist_matrix = cosine_distances(embeddings)
median_distance = float(np.median(dist_matrix))
default_eps = round(median_distance * 2.5, 3)

eps = st.slider(
    "📏 Clustering Sensitivity (DBSCAN ε)",
    min_value=0.05,
    max_value=2.0,
    step=0.05,
    value=default_eps,
    help="Lower values → more clusters (stricter similarity). Higher → fewer clusters."
)
min_samples = st.slider("👥 Minimum samples per cluster", 2, 5, 2)

st.write(f"Using eps = {eps}, min_samples = {min_samples}")

with st.spinner("Performing clustering..."):
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    cluster_labels = clusterer.fit_predict(embeddings)

num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

if num_clusters <= 1:
    st.warning("Only one cluster detected — trying more sensitive fallback (Agglomerative)...")
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6)
    cluster_labels = clusterer.fit_predict(embeddings)
    num_clusters = len(set(cluster_labels))

st.success(f"✅ Found {num_clusters} clusters (plus possible outliers).")

# --------------------------------------------------------
# Visualize with t-SNE
# --------------------------------------------------------
st.subheader("📊 Visualizing Requirement Clusters")
st.write("Reducing dimensionality for visualization...")

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=min(10, len(embeddings) - 1),
    init="pca",
    learning_rate="auto"
)
reduced = tsne.fit_transform(embeddings)

df = pd.DataFrame(reduced, columns=["x", "y"])
df["requirement"] = texts
df["cluster"] = cluster_labels

fig = px.scatter(
    df,
    x="x",
    y="y",
    color=df["cluster"].astype(str),
    text="requirement",
    title=f"Requirement Clusters (ε={eps}, clusters={num_clusters})",
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
fig.update_traces(
    textposition="top center",
    marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey"))
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# Cluster Overview
# --------------------------------------------------------
st.write("### 🧩 Cluster Overview")

cluster_map = {}
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        st.subheader("❓ Outliers (unclustered requirements)")
    else:
        st.subheader(f"Cluster {cluster_id} – Suggested Component Name:")
        suggested_name = f"Component_{cluster_id}"
        new_name = st.text_input(
            f"✏️ Name for Cluster {cluster_id}",
            value=suggested_name,
            key=f"name_{cluster_id}"
        )
        st.write(f"**Proposed Component:** {new_name}")
        cluster_map[cluster_id] = new_name

    cluster_reqs = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
    for r in cluster_reqs:
        st.markdown(f"- {r}")

# Save cluster results
st.session_state["cluster_labels"] = cluster_labels.tolist()
st.session_state["cluster_map"] = cluster_map

st.info("You can adjust ε (sensitivity) and rerun clustering to explore different architectural groupings.")
