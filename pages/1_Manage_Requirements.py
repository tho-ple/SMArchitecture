import streamlit as st
import pandas as pd
import re

st.title("🧾 Manage Requirements")

if "requirements" not in st.session_state:
    st.session_state["requirements"] = []

# --- Add Requirement Form ---
with st.expander("➕ Add New Requirement", expanded=True):
    with st.form("add_req_form", clear_on_submit=True):
        req_id = st.text_input("Requirement ID", placeholder="e.g., REQ1").strip()
        req_text = st.text_area("Requirement Description", placeholder="Describe the requirement...", height=100).strip()
        add = st.form_submit_button("Add Requirement")

    if add:
        if not req_id or not req_text:
            st.warning("Please provide both ID and description.")
        elif any(r["id"].lower() == req_id.lower() for r in st.session_state["requirements"]):
            st.error(f"ID {req_id} already exists.")
        else:
            st.session_state["requirements"].append({"id": req_id, "text": req_text})
            st.success(f"Added {req_id}")

# --- Import Options ---
st.subheader("📂 Import Requirements")
import_mode = st.radio("Import Type", ["CSV", "Gherkin (.feature)"], horizontal=True)
file = st.file_uploader("Upload file", type=["csv", "feature"])

if file:
    if import_mode == "CSV":
        df = pd.read_csv(file)
        if "requirement" in df.columns:
            new_reqs = [{"id": f"REQ{i+1}", "text": str(x)} for i, x in enumerate(df["requirement"].dropna())]
        elif {"id", "text"}.issubset(df.columns):
            new_reqs = [{"id": str(row.id), "text": str(row.text)} for _, row in df.iterrows()]
        else:
            st.error("CSV must have either 'requirement' or both 'id' and 'text' columns.")
            st.stop()
        st.session_state["requirements"].extend(new_reqs)
        st.success(f"Imported {len(new_reqs)} requirements.")
    else:
        text = file.read().decode("utf-8")
        feature = re.search(r"Feature:\s*(.*)", text)
        feature_name = feature.group(1).strip() if feature else "Unnamed Feature"
        scenarios = re.findall(r"Scenario:\s*(.*)", text)
        for i, scenario in enumerate(scenarios, start=1):
            steps = re.findall(rf"Scenario:\s*{re.escape(scenario)}[\s\S]*?(?=Scenario:|$)", text)
            step_block = steps[0] if steps else ""
            step_lines = [s.strip() for s in step_block.splitlines() if s.strip().startswith(("Given","When","Then","And","But"))]
            desc = f"{feature_name} - {scenario}: " + " ".join(step_lines)
            st.session_state["requirements"].append({"id": f"GHERKIN_{i}", "text": desc})
        st.success(f"Imported {len(scenarios)} Gherkin scenarios.")

# --- Display Table ---
if st.session_state["requirements"]:
    st.subheader("📋 Current Requirements")
    df = pd.DataFrame(st.session_state["requirements"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Remove requirement
    col1, col2 = st.columns([3, 1])
    with col1:
        rid = st.selectbox("Select requirement to remove", [""] + [r["id"] for r in st.session_state["requirements"]])
    with col2:
        if rid and st.button("🗑️ Remove"):
            st.session_state["requirements"] = [r for r in st.session_state["requirements"] if r["id"] != rid]
            st.success(f"Removed {rid}")

    # Export
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Export as CSV", data=csv, file_name="requirements.csv", mime="text/csv")
else:
    st.info("No requirements yet. Add or import some to begin.")
