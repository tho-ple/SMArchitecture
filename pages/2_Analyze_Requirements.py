# pages/2_Analyze_Requirements.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import chromadb
import os

st.set_page_config(page_title="Analyze Requirements", layout="wide")
st.title("🔍 Analyze and Cluster Requirements")

st.markdown("""
This page clusters requirements using local embeddings and several clustering algorithms.
It also shows the generated embeddings if requested.
""")

# Sidebar controls
st.sidebar.header("Settings")
embedding_model_name = st.sidebar.selectbox("Embedding model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2"])
clustering_method = st.sidebar.selectbox("Clustering algorithm", ["Agglomerative", "KMeans", "DBSCAN"])
sensitivity = st.sidebar.slider("Sensitivity (lower = broader clusters)", 0.1, 1.0, 0.4, 0.05)
show_embeddings = st.sidebar.checkbox("Show embedding vectors", False)

# Load requirements from session
if "requirements" not in st.session_state or not st.session_state.requirements:
    st.warning("No requirements found. Please add/import them in Manage Requirements first.")
    st.stop()

reqs = st.session_state.requirements
texts = [r["text"] for r in reqs]
ids = [r["id"] for r in reqs]

# Generate embeddings
st.info("Generating embeddings (SentenceTransformer)...")
model = SentenceTransformer(embedding_model_name)
embeddings = model.encode(texts, show_progress_bar=False)
embeddings = np.array(embeddings)

if embeddings.size == 0:
    st.error("Embedding generation returned no vectors.")
    st.stop()

if show_embeddings:
    st.subheader("Generated embeddings")
    df_emb = pd.DataFrame(embeddings, index=ids)
    st.dataframe(df_emb)

# Setup Chroma (local)
chroma_dir = os.path.join(os.getcwd(), "chroma_store")
os.makedirs(chroma_dir, exist_ok=True)
client = chromadb.PersistentClient(path=chroma_dir)

collection_name = "req_collection"
# get or create collection
try:
    collection = client.get_collection(collection_name)
except Exception:
    collection = client.create_collection(collection_name)

# --- Safe deletion of existing items in collection ---
try:
    # collection.get returns dict with "ids" in many versions
    existing = collection.get(include=["ids"])
    all_ids = existing.get("ids", []) if isinstance(existing, dict) else []
    # if no ids key, try collection.count() and skip deletion if zero
    if not all_ids:
        # Some chorma versions return empty lists differently; attempt a count
        try:
            if collection.count() == 0:
                all_ids = []
        except Exception:
            all_ids = []
    if all_ids:
        collection.delete(ids=all_ids)
except Exception:
    # If deletion fails for any reason, continue without breaking the UI.
    # We don't want delete semantics to block clustering.
    st.warning("Could not clear vector collection (compatibility); continuing and possibly appending.")

# Add current items
try:
    # prefer to replace: if exists remove same ids first
    # attempt to delete same ids to avoid duplicates
    try:
        collection.delete(ids=ids)
    except Exception:
        pass
    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts)
except Exception as e:
    st.warning(f"Chroma add failed: {e}")

# Choose clustering
st.info(f"Running clustering algorithm: {clustering_method}")

if clustering_method == "Agglomerative":
    # Use cosine similarity -> distance, and map sensitivity to distance_threshold
    sim = cosine_similarity(embeddings)
    dist = 1 - sim
    distance_threshold = float(np.quantile(dist, 1 - sensitivity))
    clusterer = AgglomerativeClustering(n_clusters=None,
                                        metric="cosine",
                                        linkage="average",
                                        distance_threshold=distance_threshold)
    labels = clusterer.fit_predict(embeddings)

elif clustering_method == "KMeans":
    # number of clusters derived from sensitivity and dataset size
    n_clusters = max(2, int(len(embeddings) * sensitivity))
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = clusterer.fit_predict(embeddings)

elif clustering_method == "DBSCAN":
    # DBSCAN eps heuristic: scale by median pairwise distance
    sim = cosine_similarity(embeddings)
    dist = 1 - sim
    median_dist = float(np.median(dist))
    # make eps larger for less sensitivity; inverse mapping
    eps = max(0.1, median_dist * (1.0 + (1.0 - sensitivity) * 5.0))
    clusterer = DBSCAN(eps=eps, min_samples=2, metric="cosine")
    labels = clusterer.fit_predict(embeddings)
else:
    st.error("Unknown clustering algorithm")
    st.stop()

# Build results DataFrame
df = pd.DataFrame({"id": ids, "requirement": texts, "cluster": labels})

# 2D projection for visualization
st.subheader("Cluster visualization (t-SNE)")
perplex = min(30, max(5, len(embeddings)//3))
reduced = TSNE(n_components=2, random_state=42, perplexity=perplex).fit_transform(embeddings)
df["x"], df["y"] = reduced[:, 0], reduced[:, 1]

fig = px.scatter(df, x="x", y="y", color=df["cluster"].astype(str),
                 hover_data=["id", "requirement"], title=f"Clusters ({clustering_method})")
fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
st.plotly_chart(fig, use_container_width=True)

# Show cluster lists
st.subheader("Cluster details")
for cid in sorted(df["cluster"].unique()):
    st.markdown(f"### 🧩 Cluster {cid}")
    items = df[df["cluster"] == cid][["id", "requirement"]].values.tolist()
    for i, req in items:
        st.markdown(f"- **{i}**: {req}")
    st.markdown("---")

# Download CSV
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download clusters as CSV", data=csv, file_name="clusters.csv", mime="text/csv")
