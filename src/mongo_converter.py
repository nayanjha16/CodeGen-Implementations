
def convert_to_mongo(query_plan: dict) -> str:
    """
    Converts a QueryPlan into a MongoDB shell command string (e.g. db.collection.aggregate([...]))
    """
    collection = query_plan.get("collection")
    if not collection:
        return "Error: No collection specified"

    pipeline = []

    # 1. Joins ($lookup)
    # SQL logic usually filters before joining if possible, but for simplicity we join then filter 
    # unless we can determine filters belong to the main table easily.
    # To be safe and deterministic: Match (Main Table) -> Lookup -> Unwind -> Match (Joined)
    
    # Simple strategy: separate filters for main collection vs joined fields?
    # For now, we put all filters in one $match stage at the start if they don't depend on joins,
    # or after joins if they do. 
    # Rule of thumb: If filter field has a dot (table.col), it might be joined.
    
    pre_lookup_filters = []
    post_lookup_filters = []
    
    joins = query_plan.get("joins", [])
    joined_tables = {j["table"] for j in joins}
    
    for f in query_plan.get("filters", []):
        # Heuristic: if field part before dot is in joined_tables
        parts = f["field"].split(".")
        if len(parts) > 1 and parts[0] in joined_tables:
            post_lookup_filters.append(f)
        else:
            pre_lookup_filters.append(f)
            
    # Stage 1: Initial Match
    if pre_lookup_filters:
        pipeline.append(_build_match(pre_lookup_filters))

    # Stage 2: Lookups
    for join in joins:
        # { $lookup: { from: "joined_table", localField: "...", foreignField: "...", as: "joined_table" } }
        # SQL: JOIN other ON main.x = other.y
        # We need to correctly identify which field is local and which is foreign.
        # This is tricky without schema. 
        # Heuristic: if left_on starts with collection name, it's local.
        
        lookup_stage = {
            "$lookup": {
                "from": join["table"],
                "localField": _strip_table(join["left_on"]), # simplistic
                "foreignField": _strip_table(join["right_on"]), # simplistic
                "as": join["table"]
            }
        }
        pipeline.append(lookup_stage)
        
        # SQL joins usually imply unwind (inner join semantics), but $lookup is left outer array
        # If it's INNER JOIN, we should unwind and filter out nulls?
        # For strict equivalence to SQL JOIN, we generally need $unwind
        pipeline.append({"$unwind": f"${join['table']}"})

    # Stage 3: Post-Lookup Match
    if post_lookup_filters:
        pipeline.append(_build_match(post_lookup_filters))

    # Stage 4: Grouping
    group_by = query_plan.get("group_by", [])
    aggregates = query_plan.get("aggregates", [])
    
    if group_by or aggregates:
        group_stage = _build_group(group_by, aggregates)
        pipeline.append(group_stage)
    
    # Stage 5: Sort ($sort)
    sorts = query_plan.get("sort", [])
    if sorts:
        sort_dict = {}
        for s in sorts:
            sort_dict[s["field"]] = -1 if s["order"] == "desc" else 1
        pipeline.append({"$sort": sort_dict})

    # Stage 6: Project ($project)
    # Only if NOT grouped (grouping handles projection) OR if we need to project separate from group
    # If we just grouped, the output fields are _id and accumulators. 
    # If original SQL had projections that are NOT in group/agg, that's invalid SQL usually, or implicit.
    # If we didn't group, we respect projections.
    if not group_by and not aggregates:
        projections = query_plan.get("projections", [])
        if projections and "*" not in projections:
            proj_dict = {"_id": 0}
            for p in projections:
                proj_dict[p] = 1
            pipeline.append({"$project": proj_dict})
            
    # Stage 7: Limit ($limit)
    limit = query_plan.get("limit")
    if limit is not None:
        pipeline.append({"$limit": limit})

    # Convert pipeline to string
    import json
    # Custom encoder/formatting to make it look like shell query
    pipeline_str = json.dumps(pipeline, indent=2)
    return f"db.{collection}.aggregate({pipeline_str})"

def _build_match(filters):
    match_dict = {}
    for f in filters:
        op = f["op"]
        val = f["value"]
        field = f["field"]
        
        mongo_op = None
        if op == "=":
            match_dict[field] = val
            continue
        elif op == ">": mongo_op = "$gt"
        elif op == "<": mongo_op = "$lt"
        elif op == ">=": mongo_op = "$gte"
        elif op == "<=": mongo_op = "$lte"
        elif op == "!=": mongo_op = "$ne"
        
        if mongo_op:
            if field not in match_dict:
                match_dict[field] = {}
            match_dict[field][mongo_op] = val
            
    return {"$match": match_dict}

def _build_group(group_cols, aggregates):
    group_id = {}
    if not group_cols:
        group_id = None # Aggregation on whole collection
    elif len(group_cols) == 1:
        group_id = f"${group_cols[0]}"
    else:
        for col in group_cols:
            safe_col = col.replace(".", "_") # keys can't have dots
            group_id[safe_col] = f"${col}"
            
    group_stage = {"_id": group_id}
    
    for agg in aggregates:
        func = agg["func"]
        field = agg["field"]
        alias = agg["alias"] or f"{func}_{field}"
        
        if func == "count":
            # $sum: 1
            group_stage[alias] = {"$sum": 1}
        elif func == "sum":
            group_stage[alias] = {"$sum": f"${field}"}
        elif func == "avg":
            group_stage[alias] = {"$avg": f"${field}"}
        elif func == "min":
            group_stage[alias] = {"$min": f"${field}"}
        elif func == "max":
            group_stage[alias] = {"$max": f"${field}"}
            
    return {"$group": group_stage}

    return field_name

def convert_complex_query_plan(query_plan: dict, subquery_ids: dict = None) -> str:
    """
    Complex converter that supports multi-stage pipelines (Pre-Match, Lookup, Group, Project, Post-Match).
    
    Args:
        query_plan: The parsed AST dict.
        subquery_ids: Dict mapping field names to list of IDs allowed (from execution). 
                      e.g. {'department_id': [1, 2, 5]}
    """
    collection = query_plan.get("collection")
    if not collection: return "Error: No collection"
    
    pipeline = []
    
    # 1. Pre-Aggregation Match (WHERE + Subquery IDs)
    match_dict = {}
    
    # Standard Filters
    for f in query_plan.get("filters", []):
        # We need to distinguish between pre-lookup and post-lookup filters generally
        # For simplicity in complex mode, we try to put simple filters first
        op = f["op"]
        val = f["value"]
        field = f["field"]
        
        mongo_op = None
        if op == "=": mongo_op = "$eq" # strict syntax helpful
        elif op == ">": mongo_op = "$gt"
        elif op == "<": mongo_op = "$lt"
        elif op == ">=": mongo_op = "$gte"
        elif op == "<=": mongo_op = "$lte"
        elif op == "!=": mongo_op = "$ne"
        
        if mongo_op:
            if mongo_op == "$eq":
                 match_dict[field] = val
            else:
                 if field not in match_dict: match_dict[field] = {}
                 match_dict[field][mongo_op] = val

    # Subquery Injection ($in)
    if subquery_ids:
        for field, ids in subquery_ids.items():
            if field not in match_dict: match_dict[field] = {}
            match_dict[field]["$in"] = ids
            
    if match_dict:
        pipeline.append({"$match": match_dict})
        
    # 2. Lookups (Joins)
    joins = query_plan.get("joins", [])
    for join in joins:
        pipeline.append({
            "$lookup": {
                "from": join["table"],
                "localField": _strip_table(join["left_on"]),
                "foreignField": _strip_table(join["right_on"]),
                "as": join["table"]
            }
        })
        pipeline.append({"$unwind": f"${join['table']}"})
        
    # 3. Grouping
    group_by = query_plan.get("group_by", [])
    aggregates = query_plan.get("aggregates", [])
    
    if group_by or aggregates:
        pipeline.append(_build_group(group_by, aggregates))
        
    # 4. Filter (HAVING) - Post Aggregation
    having_filters = query_plan.get("having_filters", [])
    if having_filters:
        having_dict = {}
        for f in having_filters:
            # Field here is usually like "count(*)" which maps to alias "count_*"
            # We need to find the alias used in the group stage
            # Heuristic: verify against known aggregates
            # sqlglot raw sql: "count(*)" -> we used "count_*" as default alias
            
            raw_field = f["field"].lower()
            # Simplistic mapping
            alias = raw_field.replace("(", "_").replace(")", "").replace("*", "ALL") # count(*) -> count_ALL
            # But wait, our _build_group uses `func_field` e.g. count_*
            
            # Better: Search in aggregates logic
            # If function is count and field is *, alias is count_*
            target_alias = f["field"] # Fallback
            
            # Try to map common patterns
            if "count(*)" in raw_field: target_alias = "count_*" # Matches standard
            
            op = f["op"]
            val = f["value"]
            
            mongo_op = None
            if op == ">": mongo_op = "$gt"
            elif op == "<": mongo_op = "$lt"
            elif op == ">=": mongo_op = "$gte"
            elif op == "<=": mongo_op = "$lte"
            elif op == "=": mongo_op = "$eq"
            
            if mongo_op:
                if target_alias not in having_dict: having_dict[target_alias] = {}
                having_dict[target_alias][mongo_op] = val
        
        if having_dict:
            pipeline.append({"$match": having_dict})

    # 5. Project (if needed explicit)
    # usually group handles it, but if we need specific derived fields, we add here
    
    # 6. Sort
    sorts = query_plan.get("sort", [])
    if sorts:
        sort_dict = {}
        for s in sorts:
            sort_dict[s["field"]] = -1 if s["order"] == "desc" else 1
        pipeline.append({"$sort": sort_dict})
        
    # 7. Limit
    limit = query_plan.get("limit")
    if limit is not None:
        pipeline.append({"$limit": limit})
        
    import json
    return f"db.{collection}.aggregate({json.dumps(pipeline, indent=2)})"
