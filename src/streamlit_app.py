import streamlit as st
import json
import os
import tempfile
from smart.slm import SLM
from smart.utils import UTILS
from smart.dataloading.mongodbmock import setup_mock_db
from smart.dataloading.sqldb import setup_sql_db
from smart.feeback import predict_schema

#upload files
uploaded_file = st.sidebar.file_uploader("Upload Schema JSON", type="json")
if uploaded_file:
    schema = json.load(uploaded_file)

nosqldb_file = st.sidebar.file_uploader("Upload NoSql db JSON", type="json")
sqlLite_file = st.sidebar.file_uploader("Upload SQLite db", type=None)


@st.cache_resource
def setup_databases(nosqldb_file):
    if nosqldb_file:
        return setup_mock_db(nosqldb_file)

@st.cache_resource
def get_sql_db(sqlLite_file):
    base_dir = os.path.dirname(__file__)
    if sqlLite_file:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.db', delete=False) as f:
            f.write(sqlLite_file.read())
            temp_path = f.name
        return setup_sql_db(temp_path)

@st.cache_resource
def get_mapping_data(master_data_list):
    nlq_sql_map = {}
    print(" mapping data...",master_data_list)
    for entry in master_data_list:
            print("Entry:",entry)
            # Using .lower() for a case-insensitive match
            question = entry.get("question")
            sql_query = entry.get("SQL")
            # Add to map if both fields exist
            if question and sql_query:
                nlq_sql_map[question] = sql_query
    return nlq_sql_map
# Load schema and setup DB once
db = setup_databases(nosqldb_file)
sqldb = get_sql_db(sqlLite_file)


# Initialize SLM and UTILS
slm = SLM()
utils = UTILS()

masterData = st.sidebar.file_uploader("Upload master data JSON", type="json")
if masterData:
    master_data = json.load(masterData)
else:
    master_data = []

# Now pass the variable into the function
nlqsqlmap = get_mapping_data(master_data)

st.title("SMART SLM Query Interface")

nlq = st.text_input("Enter your Natural Language Query:")



if st.button("Submit Query"):
    if nlq:
        with st.spinner("Processing query..."):
            # Try to get SQL from dev.json first
              
            sql_query = nlqsqlmap.get(nlq)
            # Generate MongoDB
            mongo_response = slm.predict_schema(nlq, schema)
            mongodb_query = utils.extract_mongodb_query(mongo_response)
            
            # Execute queries
            sql_result = None
            mongo_result = None
            
            sql_value = None
            if sql_query:
                try:
                    sql_result = sqldb.execute(sql_query).fetchone()
                    sql_value = sql_result[0] if sql_result else None
                    st.subheader("SQL Execution Result:")
                    st.write(sql_result[0])
                except Exception as e:
                    st.error(f"SQL execution failed: {e}")
            
            if mongodb_query:
                try:
                    mongoResult = utils.run_extracted_query(db, mongodb_query)
                    st.subheader("MongoDB Execution Result:")
                    st.json(mongoResult)
                except Exception as e:
                    st.error(f"MongoDB execution failed: {e}")
            resultComparison = utils.compareResults(sql_value, mongoResult)
            if resultComparison:
                st.write(f"Results are matching")
            else:
                st.write(f"Results are not matching. Executing feedback step...")
                newMongoQueryResponse = predict_schema(nlq, schema,sql_query,mongodb_query)
                updated_mongodb_query = utils.extract_mongodb_query(newMongoQueryResponse)
                mongoResult = utils.run_extracted_query(db, updated_mongodb_query)
                st.subheader("After Feed Back MongoDB Execution Result:")
                st.json(mongoResult)
                resultComparison = utils.compareResults(sql_value, mongoResult)
                st.write(f"After feedback, results match: {resultComparison}")       
    else:
        st.warning("Please enter a query.")