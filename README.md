# 🧩 SMArchitecture

**Semantic Requirement Clustering for Architectural Insights**

SMArchitecture is a Streamlit-based app that helps you **analyze functional requirements** through **embeddings** and **unsupervised clustering**, revealing potential architectural boundaries.

---

## 🚀 Features

- **Requirements Management**
  - Add requirements manually or import/export via CSV.
  - Parse Gherkin-style feature files.

- **Embedding Generation**
  - Uses local transformer models (e.g., `all-MiniLM-L6-v2`) for text embeddings.
  - No API keys or cloud services needed.

- **Clustering & Visualization**
  - Multiple clustering algorithms: `DBSCAN`, `HDBSCAN`, `Agglomerative`.
  - Adjustable sensitivity for tuning cluster granularity.
  - 2D interactive embedding visualization using Plotly.

- **Architecture Discovery**
  - Identify emerging requirement clusters.
  - Use semantic groups to derive candidate system components.

---

## ▶️ Usage

- ** Install dependencies using requirements.txt

- ** Start the app:

streamlit run Getting_Started.py


Navigate through the sidebar:

Getting Started – Learn how to use the app.

Manage Requirements – Add or import requirements.

Analyze Requirements – Generate embeddings and cluster them.


## 🗂️ Folder Structure
smarchitecture/
│
├── Getting_Started.py       # Landing / Info page
├── pages/
│   ├── 1_Manage_Requirements.py
│   └── 2_Analyze_Requirements.py
├── requirements.txt
├── .gitignore
└── README.md
