import streamlit as st
import os
from Data.Spider.SpiderLoader import SpiderLoader
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Graph.GraphBuilder import GraphBuilderFactory
from Core.Retriever.SchemaRetriever import SchemaRetriever
from Core.LLM.SQLGenerator import SQLGenerator
from Core.LLM.OllamaLLMClient import OllamaLLMClient

# Set page config
st.set_page_config(page_title="Text-to-SQL Generator", page_icon="🗄️")

st.title("🗄️ Text-to-SQL Query Generator")
st.markdown("Ask questions in natural language and get SQL queries generated!")

# Get available databases
@st.cache_data
def get_available_databases():
    db_dir = "Data/Spider/database"
    dbs = [d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))]
    return sorted(dbs)

available_dbs = get_available_databases()

# Database selection
selected_db = st.selectbox(
    "Select a database:",
    available_dbs,
    index=available_dbs.index("concert_singer") if "concert_singer" in available_dbs else 0,
    help="Choose a database schema to work with"
)

# Initialize components for selected database (cached per database)
@st.cache_resource
def initialize_components(db_id):
    spider_root = "Data/Spider"

    loader = SpiderLoader(spider_root)
    schema = loader._load_db_schema(db_id)

    fac = ChunkFactory()
    chunks = fac.schema_to_chunks(schema)

    builder_factory = GraphBuilderFactory()
    G = builder_factory.build_graph('lgraphrag', chunks)

    retriever = SchemaRetriever(G)

    llm = OllamaLLMClient()
    generator = SQLGenerator(llm)

    return retriever, generator, schema

# Initialize components when database changes
if 'current_db' not in st.session_state or st.session_state.current_db != selected_db:
    with st.spinner(f"Loading database '{selected_db}' and building knowledge graph..."):
        try:
            retriever, generator, schema = initialize_components(selected_db)
            st.session_state.current_db = selected_db
            st.session_state.retriever = retriever
            st.session_state.generator = generator
            st.session_state.schema = schema
            st.success(f"✅ Database '{selected_db}' loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load database '{selected_db}': {e}")
            st.stop()
else:
    retriever = st.session_state.retriever
    generator = st.session_state.generator
    schema = st.session_state.schema

# Show database info
st.subheader("Database Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tables", len(schema['table_names']))
with col2:
    st.metric("Columns", len(schema['column_names']))
with col3:
    st.metric("Relationships", len(schema['foreign_keys']))

# User input
question = st.text_input("Enter your question:", placeholder="e.g., What are the names of all students?")
nosql_target = st.selectbox(
    "Select target NoSQL dialect:",
    ["mongodb", "dynamodb", "cosmosdb"],
    index=0,
    help="Choose how the SQL should be translated."
)

if st.button("Generate SQL", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL query..."):
            try:
                # Retrieve relevant schema
                _, subgraph = retriever.retrieve(question)

                # Generate SQL
                sql = generator.generate_sql(question, subgraph)
                # Generate NoSQL from SQL
                nosql = generator.generate_nosql(sql, subgraph, target=nosql_target)

                # Display results
                st.success("SQL and NoSQL Generated!")
                col_sql, col_nosql = st.columns(2)
                with col_sql:
                    st.subheader("SQL")
                    st.code(sql, language="sql")
                with col_nosql:
                    st.subheader(f"NoSQL ({nosql_target})")
                    st.code(nosql, language="javascript")

            except Exception as e:
                st.error(f"❌ Error generating SQL: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by Ollama & Phi-3")