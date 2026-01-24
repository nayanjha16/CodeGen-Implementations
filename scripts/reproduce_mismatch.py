import json

def compare_results(sql_res, mongo_res):
    if len(sql_res) != len(mongo_res):
        print("Length mismatch")
        return False
        
    def normalize_item(item):
        norm = []
        keys = sorted(item.keys())
        if len(keys) == 1:
            val = item[keys[0]]
            if isinstance(val, float):
                val = round(val, 2)
            norm.append(('__single_value__', str(val)))
        else:
            for k in keys:
                val = item[k]
                if isinstance(val, float):
                    val = round(val, 2)
                norm.append((k, str(val)))
        return frozenset(norm)

    sql_set = {normalize_item(i) for i in sql_res}
    mongo_set = {normalize_item(i) for i in mongo_res}
    
    print(f"SQL Set: {sql_set}")
    print(f"MQL Set: {mongo_set}")
    
    return sql_set == mongo_set

# Simulation of the user's scenario
sql_result = [{"COUNT(*)": 17686}]
mongo_result = [{"count": 17686}]

print("Comparing...")
match = compare_results(sql_result, mongo_result)
print(f"Match: {match}")
