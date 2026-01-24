import json

def get_nlq_sql_map(data):
    """
    Loads a JSON file and returns the SQL query for a specific natural language question.
    """
    nlq_sql_map = {}

    try:
        # Load the data from the provided JSON file
        with open(data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Search for the question in the list of dictionaries
        for entry in data:
            # Using .lower() for a case-insensitive match
            question = entry.get("question")
            sql_query = entry.get("SQL")
            # Add to map if both fields exist
            if question and sql_query:
                nlq_sql_map[question] = sql_query
        return nlq_sql_map
    
    except FileNotFoundError:
        return "Error: The file was not found."
    except json.JSONDecodeError:
        return "Error: Failed to decode JSON."

def get_sql_for_nlq(nlq, target_db_id):
    """
    Returns the SQL query for an exact NLQ match from dev.json
    """
    try:
        with open('dev.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry in data:
            if entry.get("db_id") == target_db_id and entry.get("question") == nlq:
                return entry.get("SQL")
        return None
    except Exception as e:
        print(f"Error getting SQL: {e}")
        return None

""" nlqsqlmap = get_nlq_sql_map('debit_card_specializing')
print(nlqsqlmap.get('How many gas stations in CZE has Premium gas?'))
all_questions = list(nlqsqlmap.keys())
print(all_questions) """
# nlqsqlmap = get_nlq_sql_map('debit_card_specializing')
# for nlq, sql in nlqsqlmap.items():
#     print(f"NLQ: {nlq}\nSQL: {sql}\n")