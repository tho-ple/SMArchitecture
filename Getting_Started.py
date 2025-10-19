import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Getting Started",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Getting Started")
st.markdown("""
Welcome to **SMArchitecture** — a tool for **semantic analysis and clustering of software requirements** to support early **architecture discovery**.

---

### 🧭 What this app does

This app helps you:
1. **Manage and import requirements** (manually, from CSV, or Gherkin feature files).  
2. **Generate embeddings** using local language models (no OpenAI key required).  
3. **Cluster semantically similar requirements** to discover potential architectural components.  
4. **Visualize embeddings** and observe emerging groups that hint at design boundaries.

You can switch between **pages** in the sidebar:
- 🧩 **Manage Requirements:** Add, import, or export requirements.  
- 🔍 **Analyze Requirements:** Generate embeddings, cluster, and visualize results.

---

### ⚙️ How to use

1. Go to **Manage Requirements** and enter or import your functional requirements.  
2. Navigate to **Analyze Requirements**.  
3. Click **Generate Embeddings** — the app will compute vector representations for each requirement.  
4. Choose a clustering method (DBSCAN, HDBSCAN, or Agglomerative) and adjust sensitivity.  
5. Explore the 2D visualization to identify related requirement groups.  
6. Use the emerging clusters to propose **system components** or **design boundaries**.

---

### 🍕 Example: Pizza Delivery
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
