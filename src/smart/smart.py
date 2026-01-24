from slm import SLM
from tqdm import tqdm
from dataloading.mongodbmock import setup_mock_db
from dataloading.sqldb import setup_sql_db
from dataloading.sqlfinder import get_nlq_sql_map
from feeback import predict_schema
from utils import UTILS
import json
import os
import re
import ast
from itertools import islice

class SMART:
    def __init__(self, train_examples, mongo_uri, db_name):
        self.slm = SLM()
        self.utils = UTILS()
        #self.executor = MongoExecutor(mongo_uri, db_name)
    
    def run(self, nlq, schema):
        print("Starting SMART pipeline...")
        # Setup mock DB
        db = setup_mock_db('nosql_final_full.json')
        print("Mock database setup complete.",db.version)
        #set up sql database 
        sqldb = setup_sql_db('debit_card_specializing.sqlite')
        sqlResults = []
        mongoResults = []
        nlqsqlmap = get_nlq_sql_map('debit_card_specializing')
        matchingCount = 0
        totalExecutions = 0
        for nlq, sql in islice(nlqsqlmap.items(), 10):
            totalExecutions += 1
            sqlResult = sqldb.execute(sql).fetchone()
            # Normalize sqlResult to a scalar value (`sql_value`) or None
            if sqlResult is None:
                sql_value = None
            else:
                try:
                    sql_value = sqlResult[0]
                except Exception:
                    sql_value = sqlResult
            pred_schema = self.slm.predict_schema(nlq, schema)
            mongodb_query = self.utils.extract_mongodb_query(pred_schema)
            mongoResult = self.utils.run_extracted_query(db, mongodb_query)
            resultComparison = self.utils.compareResults(sql_value, mongoResult)
            if resultComparison:
                matchingCount += 1
            else:
                newMongoQueryResponse = predict_schema(nlq, schema,sql,mongodb_query)
                updated_mongodb_query = self.utils.extract_mongodb_query(newMongoQueryResponse)
                mongoResult = self.utils.run_extracted_query(db, updated_mongodb_query)
                resultComparison = self.utils.compareResults(sql_value, mongoResult)
                if resultComparison:
                    matchingCount += 1
                      
            sqlResults.append(sql_value)
            mongoResults.append(mongoResult)
        print("Query Results:", sqlResults, mongoResults)
        print(f"Total Executions: {totalExecutions}, Matching Results: {matchingCount}, Execution accuracy: {matchingCount/totalExecutions if totalExecutions > 0 else 0}")
if __name__ == "__main__":
    sm = SMART(train_examples=[], mongo_uri="mongodb://localhost:27017", db_name="testdb")
    base_dir = os.path.dirname(__file__)
    nlq_path = os.path.join(base_dir, 'data', 'NOSQL_debit_card_specializing.json')
    if not os.path.exists(nlq_path):
        alt_path = os.path.abspath(os.path.join(base_dir, '..', 'data', 'NOSQL_debit_card_specializing.json'))
        if os.path.exists(alt_path):
            nlq_path = alt_path
    try:
        with open(nlq_path, 'r', encoding='utf-8') as f:
            nlq_schema = json.load(f)
    except FileNotFoundError:
        print(f"NLQ schema file not found at {nlq_path}; proceeding without it.")
        nlq_schema = None

    test_nlqs = ["There's one customer spent 214582.17 in the June of 2013, which currency did he/she use?"]

    sm.run(test_nlqs, nlq_schema)