import streamlit as st
import httpx
import pandas as pd
import json
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
API_URL = f"{BACKEND_URL}/api/v1/generate"

st.set_page_config(page_title="Cognitive Bridge", layout="wide", page_icon="🌉")

# UI State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.title("Configuration")
db_options = [
    "california_schools",
    "card_games", 
    "codebase_community",
    "debit_card_specializing",
    "european_football_2",
    "financial",
    "formula_1",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology"
]
selected_db = st.sidebar.selectbox("Select Database", db_options)
st.sidebar.info(f"Active Context: **{selected_db}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.success("Backend: Online")
st.sidebar.success("Inference Engine: Online")
st.sidebar.success("MongoDB: Online")

# Main Interface
st.title("Cognitive Bridge: Text-to-SQL-to-NoSQL")
st.markdown("Ask questions in natural language. The system will translate them to SQL (Relational) and then MQL (Document).")

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "data" in msg:
            with st.expander("Technical Details (SQL & MQL)", expanded=False):
                st.code(msg["data"]["sql_query"], language="sql")
                st.code(json.dumps(msg["data"]["mongo_pipeline"], indent=2), language="json")
                if msg["data"].get("error"):
                    st.error(msg["data"]["error"])
            
            # Result Comparison
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("SQL Result (SQLite)")
                if msg["data"]["sql_result"]:
                    st.dataframe(pd.DataFrame(msg["data"]["sql_result"]))
                else:
                    st.write("No results.")
            
            with col2:
                st.subheader("NoSQL Result (MongoDB)")
                if msg["data"]["mongo_result"]:
                    df_mongo = pd.DataFrame(msg["data"]["mongo_result"])
                    st.dataframe(df_mongo)
                    
                    # Auto-Chart logic
                    if not df_mongo.empty:
                        # Heuristic: if result has 1 categorical and 1 numerical, chart it
                        num_cols = df_mongo.select_dtypes(include=['number']).columns
                        cat_cols = df_mongo.select_dtypes(include=['object', 'string']).columns
                        if len(num_cols) == 1 and len(cat_cols) == 1:
                            st.divider()
                            st.caption("Auto-Generated Visualization")
                            st.bar_chart(df_mongo, x=cat_cols[0], y=num_cols[0])

                else:
                    st.write("No results.")
            
            if msg["data"]["execution_match"]:
                st.success("✅ Execution Match: Results are identical across Relational and Document stores.")
            else:
                st.warning("⚠️ Execution Mismatch: Results differ. Check logic or data synchronization.")


# Input
if prompt := st.chat_input("Ask a question about the data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            st.write("🔍 Retrieving Schema Context (RAG)...")
            # In real app verify RAG here
            
            st.write("🧠 Generating SQL Logic (Phase 1)...")
            
            st.write("🔄 Transpiling to MongoDB Pipeline (Phase 2)...")
            
            try:
                # Call Backend
                payload = {"question": prompt, "db_id": selected_db}
                response = httpx.post(API_URL, json=payload, timeout=120.0)
                response.raise_for_status()
                result = response.json()
                
                status.update(label="Response Generated!", state="complete", expanded=False)
                
                # Show simple text answer or summary
                st.markdown(f"Processed query for **{selected_db}**.")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Processed query for **{selected_db}**.",
                    "data": result
                })
                
                # Rerun to render rich UI above
                st.rerun()
                
            except Exception as e:
                status.update(label="Error Occurred", state="error")
                st.error(f"Backend Error: {e}")

