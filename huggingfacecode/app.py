"""
Streamlit Frontend for NL-to-NoSQL Conversion
Docker-ready version for HuggingFace Spaces
"""

import streamlit as st
import requests
import time
import os

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="NL-to-NoSQL Converter",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CHECK BACKEND ====================

API_URL = "http://localhost:7860"

def check_backend():
    """Check if backend server is ready"""
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                if data.get("models_loaded"):
                    return True
        except:
            pass
        time.sleep(2)
    return False

# Check backend on startup
if 'backend_checked' not in st.session_state:
    with st.spinner("🔄 Starting backend server and loading models... (this may take 1-2 minutes)"):
        backend_ready = check_backend()
        st.session_state.backend_checked = True
        st.session_state.backend_ready = backend_ready
        
    if backend_ready:
        st.success("✅ Backend server ready!")
    else:
        st.error("❌ Backend server not responding. Please refresh the page.")
        st.stop()

# ==================== API CLIENT ====================

def call_api(endpoint, data):
    """Call backend API"""
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

# ==================== UI ====================

# Title
st.markdown('<h1 style="text-align: center; color: #1f77b4;">🚀 NL-to-NoSQL Conversion System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Transform Natural Language to SQL and MongoDB Queries</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")
mode = st.sidebar.radio(
    "Select Mode",
    ["🚀 Complete Pipeline", "📝 Text-to-SQL Only", "🔄 SQL-to-MongoDB Only", "🔍 Schema Translation (RAG)"]
)

# Add info about the system
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
st.sidebar.info("""
**Model**: Qwen2.5-0.5B

**Accuracy**:
- Text-to-SQL: 100%
- SQL-to-MongoDB: 95.65%

**Features**:
- SQL injection protection
- RAG-enhanced translation
- 4 operational modes
""")

# Database schemas
SCHEMAS = {
    "employees_db": "employees(id, name, department, salary, hire_date)",
    "products_db": "products(id, name, category, price, stock)",
    "users_db": "users(id, username, email, age, country)",
    "orders_db": "orders(id, customer_id, product_id, quantity, order_date, total)",
}

# ==================== MODE 1: Complete Pipeline ====================

if mode == "🚀 Complete Pipeline":
    st.header("Complete NL → SQL → MongoDB Pipeline")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_schema = st.selectbox("Select Database Schema", list(SCHEMAS.keys()))
        st.info(f"📋 Schema: {SCHEMAS[selected_schema]}")
        
        question = st.text_area(
            "Enter your question in natural language:",
            placeholder="e.g., Show all employees in IT department",
            height=100
        )
        
        if st.button("🚀 Generate Queries", type="primary"):
            if question:
                with st.spinner("Generating queries..."):
                    start_time = time.time()
                    result = call_api("/complete-pipeline", {
                        "question": question,
                        "schema": selected_schema
                    })
                    elapsed = time.time() - start_time
                
                if result:
                    with col2:
                        st.subheader("✅ Results")
                        
                        st.markdown("**SQL Query:**")
                        st.code(result["sql_query"], language="sql")
                        
                        st.markdown("**MongoDB Query:**")
                        st.code(result["mongodb_query"], language="javascript")
                        
                        st.success(f"✅ Complete pipeline executed in {elapsed:.2f}s")
            else:
                st.warning("⚠️ Please enter a question")

# ==================== MODE 2: Text-to-SQL ====================

elif mode == "📝 Text-to-SQL Only":
    st.header("Natural Language to SQL Conversion")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_schema = st.selectbox("Select Database Schema", list(SCHEMAS.keys()))
        st.info(f"📋 Schema: {SCHEMAS[selected_schema]}")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., Show products with price greater than 100",
            height=150
        )
        
        if st.button("Generate SQL Query", type="primary"):
            if question:
                with st.spinner("Generating SQL query..."):
                    start_time = time.time()
                    result = call_api("/text-to-sql", {
                        "question": question,
                        "schema": selected_schema
                    })
                    elapsed = time.time() - start_time
                
                if result:
                    with col2:
                        st.subheader("✅ Generated SQL Query")
                        st.code(result["sql_query"], language="sql")
                        st.caption(f"⏱️ Generated in {elapsed:.2f}s")
                        st.success("✅ SQL query generated successfully!")
            else:
                st.warning("⚠️ Please enter a question")

# ==================== MODE 3: SQL-to-MongoDB ====================

elif mode == "🔄 SQL-to-MongoDB Only":
    st.header("SQL to MongoDB Query Conversion")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        sql_input = st.text_area(
            "Enter SQL Query:",
            placeholder="SELECT * FROM users WHERE age > 25",
            height=150
        )
        
        if st.button("Convert to MongoDB", type="primary"):
            if sql_input:
                with st.spinner("Converting to MongoDB..."):
                    start_time = time.time()
                    result = call_api("/sql-to-mongodb", {
                        "sql_query": sql_input
                    })
                    elapsed = time.time() - start_time
                
                if result:
                    with col2:
                        st.subheader("✅ MongoDB Query")
                        st.code(result["mongodb_query"], language="javascript")
                        st.caption(f"⏱️ Converted in {elapsed:.2f}s")
                        st.success("✅ Conversion successful!")
            else:
                st.warning("⚠️ Please enter a SQL query")

# ==================== MODE 4: Schema Translation ====================

elif mode == "🔍 Schema Translation (RAG)":
    st.header("SQL Schema to MongoDB Translation (RAG-Enhanced)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        sql_schema = st.text_area(
            "Enter SQL CREATE TABLE statement:",
            placeholder="CREATE TABLE users (\n  id INT PRIMARY KEY,\n  name VARCHAR(100),\n  email VARCHAR(100)\n)",
            height=200
        )
        
        use_rag = st.checkbox("Use RAG (Retrieval-Augmented Generation)", value=True)
        k = st.slider("Number of similar examples to retrieve", 1, 5, 3) if use_rag else 0
        
        if st.button("Translate Schema", type="primary"):
            if sql_schema:
                with st.spinner("Translating schema..."):
                    start_time = time.time()
                    result = call_api("/schema-translation", {
                        "sql_schema": sql_schema,
                        "use_rag": use_rag,
                        "k": k
                    })
                    elapsed = time.time() - start_time
                
                if result:
                    with col2:
                        st.subheader("✅ MongoDB Schema")
                        st.json(result["mongodb_schema"])
                        st.caption(f"⏱️ Generated in {elapsed:.2f}s")
                        
                        if use_rag and result.get("similar_examples"):
                            st.markdown("---")
                            st.subheader("📚 Similar Examples Retrieved")
                            for i, ex in enumerate(result["similar_examples"], 1):
                                st.write(f"**{i}. {ex['name']}** - Similarity: {ex['similarity']*100:.1f}%")
            else:
                st.warning("⚠️ Please enter a SQL schema")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>NL-to-NoSQL Conversion System</strong></p>
    <p>Powered by Qwen2.5-0.5B + QLoRA | 100% Text-to-SQL | 95.65% SQL-to-MongoDB Accuracy</p>
    <p style='font-size: 12px;'>Models: 1.2 GB RAM | Inference: <2s | Training: 25 min on free Colab</p>
</div>
""", unsafe_allow_html=True)
