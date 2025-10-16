import streamlit as st
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Requirement Analyzer",
    page_icon="🔍",
    layout="wide",
)

# ---------- HEADER ----------
st.title("🔍 Requirement Analyzer")

st.markdown("""
Welcome to the **Requirement Analyzer** — a tool to help you **analyze and structure software requirements** 
based on their semantic similarity.

Instead of manually grouping requirements, this app uses **language embeddings** and **automatic clustering** to reveal
emerging **themes or design areas**.  
These clusters can guide early **architectural decisions**, such as identifying modules, subsystems, or bounded contexts.
""")

st.divider()

# ---------- HOW IT WORKS ----------
st.header("⚙️ How It Works")

st.markdown("""
1. **Input Requirements**  
   Add or import your requirements via the **🧾 Manage Requirements** page.  
   You can also load an example dataset below.

2. **Embedding Generation & Vector Storage**  
   The app converts each requirement into a high-dimensional vector using a **Sentence Transformer model**.  
   These embeddings are stored locally in **ChromaDB**, a lightweight, persistent vector database.

3. **Semantic Clustering**  
   ChromaDB groups semantically similar requirements using **vector similarity search**.  
   Each group represents a **potential component** or **bounded context**.

4. **Naming the Components**  
   The app analyzes key terms in each cluster to **suggest a name** —  
   which you can refine to describe the architecture's emerging structure.
""")

st.success("This workflow mirrors the EarlyBird design thinking process: from text → semantics → structure.")

st.divider()

# ---------- GETTING STARTED ----------
st.header("🚀 Getting Started")

st.markdown("""
### Step-by-step

1. **Go to “🧾 Manage Requirements”** in the sidebar.  
   - Add requirements manually  
   - Or import them from a CSV or Gherkin `.feature` file  
2. **Then open “📊 Analyze Requirements.”**  
   - Select an embedding model  
   - Adjust the *minimum cluster size* (optional)  
   - Click **Run Analysis**  
3. Review the discovered clusters, rename them meaningfully, and export results to CSV.

### Why use clustering for design?
- Groups of related requirements often correspond to **potential architectural modules**
- Helps identify **redundant** or **conflicting** requirements
- Reveals **natural domains** in complex systems
""")

st.divider()

# ---------- EXAMPLE ----------
st.header("🍕 Example: Pizza Delivery App Requirements")

st.markdown("""
Here’s an example set of mixed functional and non-functional requirements for a fictional pizza delivery system.  
You can load these into the app and run the analysis immediately.
""")

example_reqs = [
    # --- Functional ---
    {"id": "REQ1", "text": "Users should be able to order pizzas with multiple toppings via a web or mobile interface."},
    {"id": "REQ2", "text": "The system must support online payment using credit cards, Apple Pay, and PayPal."},
    {"id": "REQ3", "text": "Deliveries must be tracked in real time by both the customer and the driver."},
    {"id": "REQ4", "text": "Administrators should be able to update menu items, categories, and prices dynamically."},
    {"id": "REQ5", "text": "Customers should receive notifications when their order is confirmed, baked, and dispatched."},
    {"id": "REQ6", "text": "The system should provide estimated delivery times based on driver location and traffic."},
    {"id": "REQ7", "text": "The platform should support promotional codes and loyalty discounts for frequent customers."},
    {"id": "REQ8", "text": "The customer should be able to rate and review their delivery experience."},
    {"id": "REQ9", "text": "Menu data should be cached locally to minimize load times for returning users."},
    {"id": "REQ10", "text": "The system should allow administrators to temporarily disable delivery during maintenance windows."},
    # --- Non-functional ---
    {"id": "REQ11", "text": "All transactions must be encrypted using TLS 1.3 or higher."},
    {"id": "REQ12", "text": "The system should handle up to 10,000 concurrent users without performance degradation."},
    {"id": "REQ13", "text": "The application should maintain 99.9% uptime per month."},
    {"id": "REQ14", "text": "The system should comply with GDPR and store customer data securely."},
    {"id": "REQ15", "text": "The UI should be fully responsive and accessible according to WCAG 2.1 AA standards."},
    {"id": "REQ16", "text": "Error messages should be descriptive and help users recover without support intervention."},
    {"id": "REQ17", "text": "The system should support deployment to both cloud and on-premise environments."},
]

df = pd.DataFrame(example_reqs)
st.dataframe(df, use_container_width=True, hide_index=True)

if st.button("📥 Load Example into Session"):
    st.session_state["requirements"] = example_reqs
    st.success("✅ Example requirements loaded! You can now go to **📊 Analyze Requirements** to explore clusters.")

st.divider()
st.info("👈 Use the sidebar to navigate between pages.")
