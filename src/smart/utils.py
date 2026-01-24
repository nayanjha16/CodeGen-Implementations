import json

import re

class UTILS:
    def __init__(self):
        pass    
    def extract_mongodb_query(self,response):
        try:
            # 1. Access the text from the first candidate's first part
            raw_text = response.candidates[0].content.parts[0].text
            
            # 2. Clean the markdown code blocks if they exist
            # This removes ```json ... ``` wrappers
            json_str = re.sub(r"```json\n?|```", "", raw_text).strip()
            
            # 3. Parse the string into a dictionary
            data = json.loads(json_str)
            
            # 4. Return the specific mongodb_query key
            return data.get("mongodb_query")
        
        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            print(f"Error extracting query: {e}")
            return None
        
    def extract_sql_query(self, response):
        try:
            raw_text = response.candidates[0].content.parts[0].text
            # Clean markdown if present
            sql_str = re.sub(r"```sql\n?|```", "", raw_text).strip()
            return sql_str
        except (AttributeError, IndexError) as e:
            print(f"Error extracting SQL: {e}")
            return None
        
    def run_extracted_query(self,db,extracted_data):
        if not extracted_data:
            print("No query data provided.")
            return

        # Extract collection name and the pipeline array
        collection_name = extracted_data.get("collection")
        pipeline = extracted_data.get("pipeline")
        
        # 2. Access the specific collection
        collection = db[collection_name]
        
        # 3. Execute the aggregation pipeline
        try:
            results = list(collection.aggregate(pipeline))
            
            print(f"Executed query on collection: {collection_name}")
            return results
        except Exception as e:
            print(f"An error occurred during execution: {e}")
            return None
        
    def compareResults(self, sqlResult, mongoResult):
        # 1. Normalize mongoResult into a list for consistent iteration
        if isinstance(mongoResult, dict):
            # Extract values if it's a single dictionary
            candidates = list(mongoResult.values())
        elif isinstance(mongoResult, list):
            # Flatten internal dicts to their values if necessary
            candidates = []
            for item in mongoResult:
                if isinstance(item, dict):
                    candidates.extend(item.values())
                else:
                    candidates.append(item)
        else:
            # Wrap single scalar value in a list
            candidates = [mongoResult]
        print("Candidates extracted from MongoDB result:", candidates)
        # 2. Check if sqlResult exists anywhere in the processed candidates
        # This uses 'any' for efficiency (stops at the first match found)
        return any(self.safe_round(sqlResult) == self.safe_round(val) for val in candidates)
    
    def safe_round(self, val):
        return round(val, 2) if isinstance(val, (int, float)) else val
        
if __name__ == "__main__":
    u = UTILS()
    resultComparison = u.compareResults(47273, [{'TotalConsumption': 0.74, 'CustomerID': 47273}])
    print("Comparison Result:", resultComparison)  # Expected: True