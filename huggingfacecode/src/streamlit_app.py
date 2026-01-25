"""
Streamlit Frontend for NL-to-NoSQL Conversion
Updated for HuggingFace Spaces with src/streamlit_app.py structure
"""

import streamlit as st
import requests
import time
import json

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="NL-to-NoSQL Converter",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== BACKEND CONFIGURATION ====================

# Backend runs on port 8000 in container
# Streamlit runs on port 7860 (HuggingFace standard)
BACKEND_URL = "http://localhost:8000"

# ==================== CHECK BACKEND ====================

def check_backend():
    """Check if backend server is ready with retries"""
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=3)
            if response.status_code == 200:
                data = response.json()
                if data.get("models_loaded"):
                    return True, "Backend ready!"
        except requests.exceptions.RequestException as e:
            pass
        time.sleep(2)
    return False, "Backend timeout"

# Check backend on first load
if 'backend_checked' not in st.session_state:
    with st.spinner("🔄 Loading models... This may take 1-2 minutes on first startup."):
        backend_ready, message = check_backend()
        st.session_state.backend_checked = True
        st.session_state.backend_ready = backend_ready
        
    if backend_ready:
        st.success("✅ Backend ready! Models loaded successfully.")
    else:
        st.error(f"❌ Backend not responding: {message}")
        st.info("🔄 This may be a temporary issue. Please refresh the page.")
        st.info("If problem persists, check Space logs for errors.")
        st.stop()

# ==================== API CLIENT ====================

def call_api(endpoint, data):
    """Call backend API with error handling"""
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}", 
            json=data, 
            timeout=60
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, "⏱️ Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                error_msg = error_data.get('detail', error_msg)
            except:
                pass
        return None, f"❌ API Error: {error_msg}"

# ==================== UI ====================

# Title with custom styling
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #1f77b4; margin-bottom: 10px;'>🚀 NL-to-NoSQL Conversion System</h1>
    <p style='color: #666; font-size: 18px;'>Transform Natural Language to SQL and MongoDB Queries</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")
mode = st.sidebar.radio(
    "Select Mode",
    [
        "🚀 Complete Pipeline",
        "📝 Text-to-SQL Only",
        "🔄 SQL-to-MongoDB Only",
        "🔍 Schema Translation (RAG)"
    ],
    index=0
)

# System info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
st.sidebar.info("""
**Model**: Qwen2.5-0.5B-Instruct

**Accuracy**:
- Text-to-SQL: 100%
- SQL-to-MongoDB: 95.65%

**Features**:
- ✅ SQL injection protection
- ✅ RAG-enhanced translation
- ✅ 4 operational modes
- ✅ Security validation

**Performance**:
- Memory: ~1.2 GB
- Query time: <2s
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("[📖 Documentation](https://github.com/YOUR_USERNAME/nl-to-nosql)")
st.sidebar.markdown("[🤗 Model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)")

# Database schemas
SCHEMAS = {
    "employees_db": "employees(id, name, department, salary, hire_date)",
    "products_db": "products(id, name, category, price, stock)",
    "users_db": "users(id, username, email, age, country)",
    "orders_db": "orders(id, customer_id, product_id, quantity, order_date, total)",
    "students_db": "students(id, name, major, gpa, enrollment_year)",
}

# ==================== MODE 1: Complete Pipeline ====================

if mode == "🚀 Complete Pipeline":
    st.header("Complete NL → SQL → MongoDB Pipeline")
    st.markdown("Convert natural language questions to both SQL and MongoDB queries in one step.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input")
        
        selected_schema = st.selectbox(
            "Select Database Schema",
            list(SCHEMAS.keys()),
            help="Choose the database schema for your query"
        )
        st.info(f"📋 Schema: `{SCHEMAS[selected_schema]}`")
        
        question = st.text_area(
            "Enter your question in natural language:",
            placeholder="Example: Show all employees in IT department with salary greater than 50000",
            height=120,
            help="Ask a question about the selected database"
        )
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_button = st.button("🚀 Generate Queries", type="primary", use_container_width=True)
        with col_b:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()
        
        if run_button:
            if question:
                with st.spinner("🔄 Generating queries..."):
                    start_time = time.time()
                    result, error = call_api("/complete-pipeline", {
                        "question": question,
                        "schema": selected_schema
                    })
                    elapsed = time.time() - start_time
                
                if error:
                    st.error(error)
                elif result:
                    with col2:
                        st.subheader("✅ Results")
                        
                        st.markdown("**🔹 SQL Query:**")
                        st.code(result.get("sql_query", "Error generating SQL"), language="sql")
                        
                        st.markdown("**🔹 MongoDB Query:**")
                        st.code(result.get("mongodb_query", "Error generating MongoDB"), language="javascript")
                        
                        st.success(f"✅ Pipeline completed in {elapsed:.2f}s")
                        
                        # Show additional info if available
                        if result.get("security_status"):
                            with st.expander("🔒 Security Status"):
                                st.json(result["security_status"])
            else:
                st.warning("⚠️ Please enter a question")

# ==================== MODE 2: Text-to-SQL ====================

elif mode == "📝 Text-to-SQL Only":
    st.header("Natural Language to SQL Conversion")
    st.markdown("Convert natural language questions to SQL queries with security validation.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input")
        
        selected_schema = st.selectbox(
            "Select Database Schema",
            list(SCHEMAS.keys())
        )
        st.info(f"📋 Schema: `{SCHEMAS[selected_schema]}`")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="Example: Find all products with price greater than 100 ordered by price",
            height=150
        )
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_button = st.button("Generate SQL", type="primary", use_container_width=True)
        with col_b:
            if st.button("Clear", use_container_width=True):
                st.rerun()
        
        if run_button:
            if question:
                with st.spinner("🔄 Generating SQL query..."):
                    start_time = time.time()
                    result, error = call_api("/text-to-sql", {
                        "question": question,
                        "schema": selected_schema
                    })
                    elapsed = time.time() - start_time
                
                if error:
                    st.error(error)
                elif result:
                    with col2:
                        st.subheader("✅ Generated SQL Query")
                        
                        sql_query = result.get("sql_query", "")
                        st.code(sql_query, language="sql")
                        
                        st.caption(f"⏱️ Generated in {elapsed:.2f}s")
                        
                        # Security status
                        if result.get("security"):
                            security = result["security"]
                            if security.get("is_safe"):
                                st.success("🔒 Security: PASSED")
                            else:
                                st.warning(f"⚠️ Security: {security.get('message', 'Failed')}")
                        
                        st.success("✅ SQL query generated successfully!")
            else:
                st.warning("⚠️ Please enter a question")

# ==================== MODE 3: SQL-to-MongoDB ====================

elif mode == "🔄 SQL-to-MongoDB Only":
    st.header("SQL to MongoDB Query Conversion")
    st.markdown("Convert SQL queries to equivalent MongoDB queries.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input")
        
        sql_input = st.text_area(
            "Enter SQL Query:",
            placeholder="SELECT * FROM users WHERE age > 25 AND country = 'USA' ORDER BY age DESC LIMIT 10",
            height=200
        )
        
        # Example queries
        with st.expander("📚 Example Queries"):
            st.code("SELECT * FROM users WHERE age > 25", language="sql")
            st.code("SELECT name, COUNT(*) FROM products GROUP BY category", language="sql")
            st.code("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id", language="sql")
        
        col_a, col_b = st.columns([1, 3])
        with col_a:
            run_button = st.button("Convert", type="primary", use_container_width=True)
        with col_b:
            if st.button("Clear", use_container_width=True):
                st.rerun()
        
        if run_button:
            if sql_input:
                with st.spinner("🔄 Converting to MongoDB..."):
                    start_time = time.time()
                    result, error = call_api("/sql-to-mongodb", {
                        "sql_query": sql_input
                    })
                    elapsed = time.time() - start_time
                
                if error:
                    st.error(error)
                elif result:
                    with col2:
                        st.subheader("✅ MongoDB Query")
                        
                        mongodb_query = result.get("mongodb_query", "")
                        st.code(mongodb_query, language="javascript")
                        
                        st.caption(f"⏱️ Converted in {elapsed:.2f}s")
                        
                        # Show query type
                        if result.get("query_type"):
                            st.info(f"Query Type: `{result['query_type']}`")
                        
                        st.success("✅ Conversion successful!")
            else:
                st.warning("⚠️ Please enter a SQL query")

# ==================== MODE 4: Schema Translation ====================

elif mode == "🔍 Schema Translation (RAG)":
    st.header("SQL Schema to MongoDB Translation")
    st.markdown("Translate SQL CREATE TABLE statements to MongoDB schemas with RAG enhancement.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input")
        
        sql_schema = st.text_area(
            "Enter SQL CREATE TABLE statement:",
            placeholder="""CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    age INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);""",
            height=250
        )
        
        use_rag = st.checkbox(
            "Use RAG (Retrieval-Augmented Generation)",
            value=True,
            help="Retrieve similar examples to improve translation quality"
        )
        
        if use_rag:
            k = st.slider(
                "Number of similar examples to retrieve",
                min_value=1,
                max_value=5,
                value=3,
                help="More examples may improve quality but increase processing time"
            )
        else:
            k = 0
        
        run_button = st.button("🔍 Translate Schema", type="primary", use_container_width=True)
        
        if run_button:
            if sql_schema:
                with st.spinner("🔄 Translating schema..."):
                    start_time = time.time()
                    result, error = call_api("/schema-translation", {
                        "sql_schema": sql_schema,
                        "use_rag": use_rag,
                        "k": k
                    })
                    elapsed = time.time() - start_time
                
                if error:
                    st.error(error)
                elif result:
                    with col2:
                        st.subheader("✅ MongoDB Schema")
                        
                        mongodb_schema = result.get("mongodb_schema", {})
                        st.json(mongodb_schema)
                        
                        st.caption(f"⏱️ Generated in {elapsed:.2f}s")
                        
                        # Show similar examples if RAG was used
                        if use_rag and result.get("similar_examples"):
                            st.markdown("---")
                            st.subheader("📚 Similar Examples Retrieved")
                            
                            for i, ex in enumerate(result["similar_examples"], 1):
                                similarity_pct = ex.get('similarity', 0) * 100
                                st.write(f"**{i}. {ex.get('name', 'Unknown')}** - Similarity: {similarity_pct:.1f}%")
                                
                                with st.expander(f"View Example {i}"):
                                    st.markdown("**SQL:**")
                                    st.code(ex.get('sql', ''), language="sql")
                                    st.markdown("**MongoDB:**")
                                    st.json(ex.get('mongodb', {}))
                        
                        st.success("✅ Schema translation complete!")
            else:
                st.warning("⚠️ Please enter a SQL schema")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-weight: bold; font-size: 16px;'>NL-to-NoSQL Conversion System</p>
    <p style='font-size: 14px;'>Powered by Qwen2.5-0.5B + QLoRA Fine-tuning</p>
    <p style='font-size: 12px;'>
        📊 100% Text-to-SQL Accuracy | 95.65% SQL-to-MongoDB Accuracy<br>
        💾 1.2 GB RAM | ⚡ <2s Query Time | 🎓 Trained on Spider + BIRD datasets
    </p>
    <p style='font-size: 11px; color: #999; margin-top: 10px;'>
        Thesis Project 2025 | Zero-Cost Training on Google Colab
    </p>
</div>
""", unsafe_allow_html=True)

# Debug info in sidebar (expandable)
with st.sidebar.expander("🔧 Debug Info"):
    st.write(f"Backend URL: {BACKEND_URL}")
    st.write(f"Backend Status: {'✅ Ready' if st.session_state.get('backend_ready') else '❌ Not Ready'}")
    
    if st.button("Test Backend Connection"):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Backend responding")
                st.json(response.json())
            else:
                st.error(f"❌ Backend returned {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
