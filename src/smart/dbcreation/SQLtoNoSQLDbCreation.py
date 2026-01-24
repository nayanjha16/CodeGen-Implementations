import json

def sql_to_nosql_converter(input_file, output_file):
    # Load the flat SQL-style JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. DENORMALIZE CUSTOMERS (Embedding Consumption)
    # Map customers by ID for easy access
    customers_nosql = {}
    for c in data['customers']:
        customers_nosql[c['CustomerID']] = {
            "CustomerID": c['CustomerID'],
            "Segment": c['Segment'],
            "Currency": c['Currency'],
            "yearmonths": [] # Initialize empty array for embedding
        }

    # Embed consumption records into their respective customer documents
    for con in data['yearmonth']:
        cust_id = con['CustomerID']
        if cust_id in customers_nosql:
            customers_nosql[cust_id]['yearmonths'].append({
                "Date": con['Date'],
                "Consumption": con['Consumption']
            })

    # 2. TRANSFORM PRODUCTS (Reference Collection)
    products_nosql = [
        {"ProductID": p['ProductID'], "Description": p['Description']} 
        for p in data['products']
    ]

    # 3. TRANSFORM TRANSACTIONS (Linked Collection)
    transactions_nosql = [
        {
            "TransactionID": t['TransactionID'],
            "Date": t['Date'],
            "CustomerID": t['CustomerID'],
            "ProductID": t['ProductID'],
            "Price": t['Price']
        }
        for t in data['transactions_1k']
    ]

    # This part of the script specifically handles the gas stations
    gasstations_collection = [
        {
            "GasStationID": g['GasStationID'],
            "ChainID": g['ChainID'],
            "Country": g['Country'],
            "Segment": g['Segment']
        }
        for g in data['gasstations'] # This loops through ALL 100 stations
   ]
    # Combine into a single NoSQL Document Structure
    final_nosql = {
        "CustomersCollection": list(customers_nosql.values()),
        "ProductsCollection": products_nosql,
        "TransactionsCollection": transactions_nosql,
        "GasstationsCollection": gasstations_collection
    }

    # Save the output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_nosql, f, indent=4)

# Execute
sql_to_nosql_converter('debit_card_specializing_sql_data.json', 'nosql_final_full.json')