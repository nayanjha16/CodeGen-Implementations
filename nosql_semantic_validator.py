"""
Semantic NoSQL Validation - Validates generated NoSQL based on SQL logic
Checks if the NoSQL correctly implements the SQL query's intent
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum


class ValidationLevel(Enum):
    """Validation result levels"""
    CORRECT = "CORRECT"
    MOSTLY_CORRECT = "MOSTLY_CORRECT"
    MINOR_ISSUES = "MINOR_ISSUES"
    MAJOR_ISSUES = "MAJOR_ISSUES"
    INCORRECT = "INCORRECT"


class NoSQLSemanticValidator:
    """
    Validates generated NoSQL queries based on SQL semantics
    """
    
    def __init__(self):
        self.results = []
    
    def parse_sql(self, sql: str) -> Dict[str, Any]:
        """
        Extract semantic components from SQL query
        """
        sql_lower = sql.lower().strip()
        
        components = {
            'select_columns': [],
            'from_table': None,
            'where_conditions': [],
            'join_tables': [],
            'group_by': [],
            'order_by': [],
            'order_direction': None,
            'limit': None,
            'aggregations': [],
            'distinct': False,
        }
        
        # Extract SELECT columns
        select_match = re.search(r'select\s+(.*?)\s+from', sql_lower, re.DOTALL)
        if select_match:
            select_part = select_match.group(1).strip()
            components['distinct'] = select_part.startswith('distinct')
            if components['distinct']:
                select_part = select_part.replace('distinct', '').strip()
            
            # Check for aggregations
            components['aggregations'] = re.findall(r'(count|sum|avg|min|max)\s*\(', select_part)
            
            # Extract column names (simplified)
            cols = [c.strip() for c in select_part.split(',')]
            components['select_columns'] = cols
        
        # Extract FROM table
        from_match = re.search(r'from\s+(\w+)', sql_lower)
        if from_match:
            components['from_table'] = from_match.group(1)
        
        # Extract WHERE conditions
        where_match = re.search(r'where\s+(.*?)(?:\s+group by|\s+order by|\s+limit|$)', sql_lower, re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            components['where_conditions'] = self._extract_conditions(where_clause)
        
        # Extract JOINs
        join_matches = re.findall(r'join\s+(\w+)', sql_lower)
        components['join_tables'] = join_matches
        
        # Extract GROUP BY
        group_match = re.search(r'group by\s+(.*?)(?:\s+order by|\s+limit|$)', sql_lower)
        if group_match:
            components['group_by'] = [g.strip() for g in group_match.group(1).split(',')]
        
        # Extract ORDER BY
        order_match = re.search(r'order by\s+(.*?)(?:\s+limit|$)', sql_lower)
        if order_match:
            order_part = order_match.group(1).strip()
            components['order_direction'] = 'desc' if 'desc' in order_part else 'asc'
            # Extract column
            order_col = re.sub(r'\s+(asc|desc)', '', order_part).strip()
            components['order_by'] = [order_col]
        
        # Extract LIMIT
        limit_match = re.search(r'limit\s+(\d+)', sql_lower)
        if limit_match:
            components['limit'] = int(limit_match.group(1))
        
        return components
    
    def _extract_conditions(self, where_clause: str) -> List[Dict]:
        """Extract individual conditions from WHERE clause"""
        conditions = []
        
        # Simple patterns for common conditions
        patterns = [
            (r'(\w+)\s*=\s*[\'"]?([^\s\'"]+)[\'"]?', 'equals'),
            (r'(\w+)\s*>\s*(\d+)', 'greater_than'),
            (r'(\w+)\s*<\s*(\d+)', 'less_than'),
            (r'(\w+)\s*>=\s*(\d+)', 'greater_equal'),
            (r'(\w+)\s*<=\s*(\d+)', 'less_equal'),
            (r'(\w+)\s+like\s+[\'"]([^\'"]+)[\'"]', 'like'),
            (r'(\w+)\s+in\s+\((.*?)\)', 'in'),
        ]
        
        for pattern, cond_type in patterns:
            matches = re.findall(pattern, where_clause, re.IGNORECASE)
            for match in matches:
                conditions.append({
                    'field': match[0],
                    'type': cond_type,
                    'value': match[1] if len(match) > 1 else None
                })
        
        return conditions
    
    def parse_nosql(self, nosql: str) -> Dict[str, Any]:
        """
        Extract semantic components from NoSQL query
        """
        components = {
            'collection': None,
            'filter_fields': [],
            'projection_fields': [],
            'sort_fields': [],
            'sort_direction': None,
            'limit': None,
            'aggregation_pipeline': False,
            'has_group': False,
        }
        
        nosql_lower = nosql.lower()
        
        # Try to parse as JSON
        try:
            # Clean the string
            nosql_clean = nosql.strip()
            if not nosql_clean.startswith('{'):
                # Try to extract JSON from the string
                json_match = re.search(r'\{.*\}', nosql, re.DOTALL)
                if json_match:
                    nosql_clean = json_match.group(0)
            
            data = json.loads(nosql_clean)
            
            if isinstance(data, dict):
                components['collection'] = data.get('collection')
                
                # Extract filter fields
                if 'filter' in data and data['filter']:
                    components['filter_fields'] = list(data['filter'].keys())
                
                # Extract projection fields
                if 'projection' in data and data['projection']:
                    components['projection_fields'] = [
                        k for k, v in data['projection'].items() 
                        if v == 1 and k != '_id'
                    ]
                
                # Extract sort
                if 'sort' in data and data['sort']:
                    components['sort_fields'] = list(data['sort'].keys())
                    # Check direction (-1 = desc, 1 = asc)
                    first_sort_val = list(data['sort'].values())[0]
                    components['sort_direction'] = 'desc' if first_sort_val == -1 else 'asc'
                
                # Extract limit
                if 'limit' in data:
                    components['limit'] = data['limit']
                
                # Check for aggregation
                if 'pipeline' in data or '$group' in nosql_lower:
                    components['aggregation_pipeline'] = True
                if '$group' in nosql_lower:
                    components['has_group'] = True
        
        except json.JSONDecodeError:
            # Parse as MongoDB shell syntax
            if 'db.' in nosql_lower:
                coll_match = re.search(r'db\.(\w+)', nosql_lower)
                if coll_match:
                    components['collection'] = coll_match.group(1)
            
            # Check for filter/match
            if '$match' in nosql_lower or 'find(' in nosql_lower:
                components['filter_fields'] = re.findall(r'["\']?(\w+)["\']?\s*:', nosql_lower)
            
            # Check for projection
            if '$project' in nosql_lower:
                proj_match = re.search(r'\$project["\']?\s*:\s*\{([^}]+)\}', nosql_lower)
                if proj_match:
                    components['projection_fields'] = re.findall(r'["\']?(\w+)["\']?\s*:\s*1', proj_match.group(1))
            
            # Check for sort
            if '$sort' in nosql_lower or 'sort(' in nosql_lower:
                components['sort_fields'] = ['exists']
                if '-1' in nosql or 'desc' in nosql_lower:
                    components['sort_direction'] = 'desc'
                else:
                    components['sort_direction'] = 'asc'
            
            # Check for limit
            if '$limit' in nosql_lower or 'limit(' in nosql_lower:
                limit_match = re.search(r'limit["\']?\s*:?\s*(\d+)', nosql_lower)
                if limit_match:
                    components['limit'] = int(limit_match.group(1))
            
            # Check for aggregation
            if 'aggregate(' in nosql_lower or '$group' in nosql_lower:
                components['aggregation_pipeline'] = True
            if '$group' in nosql_lower:
                components['has_group'] = True
        
        return components
    
    def validate_translation(self, sql: str, nosql: str, index: int = 0) -> Dict[str, Any]:
        """
        Validate if NoSQL correctly implements SQL logic
        """
        sql_components = self.parse_sql(sql)
        nosql_components = self.parse_nosql(nosql)
        
        issues = []
        validations = {
            'collection_correct': True,
            'has_filter': True,
            'has_projection': True,
            'has_sort': True,
            'sort_direction_correct': True,
            'limit_correct': True,
            'handles_aggregation': True,
            'handles_group_by': True,
            'handles_joins': True,
        }
        
        # Validate collection name
        if sql_components['from_table'] and nosql_components['collection']:
            sql_table = sql_components['from_table']
            nosql_coll = nosql_components['collection']
            # Check if they match (allowing plural/singular variations)
            if sql_table not in nosql_coll and nosql_coll not in sql_table:
                validations['collection_correct'] = False
                issues.append(f"Collection mismatch: SQL uses '{sql_table}', NoSQL uses '{nosql_coll}'")
        
        # Validate WHERE -> filter
        if sql_components['where_conditions']:
            if not nosql_components['filter_fields']:
                validations['has_filter'] = False
                issues.append("SQL has WHERE clause but NoSQL has no filter")
        
        # Validate SELECT -> projection
        if sql_components['select_columns'] and '*' not in sql_components['select_columns'][0]:
            if not nosql_components['projection_fields'] and not any(agg in sql for agg in ['count', 'sum', 'avg']):
                validations['has_projection'] = False
                issues.append("SQL selects specific columns but NoSQL has no projection")
        
        # Validate ORDER BY -> sort
        if sql_components['order_by']:
            if not nosql_components['sort_fields']:
                validations['has_sort'] = False
                issues.append("SQL has ORDER BY but NoSQL has no sort")
            elif nosql_components['sort_direction'] != sql_components['order_direction']:
                validations['sort_direction_correct'] = False
                issues.append(f"Sort direction mismatch: SQL is {sql_components['order_direction']}, NoSQL is {nosql_components['sort_direction']}")
        
        # Validate LIMIT
        if sql_components['limit']:
            if nosql_components['limit'] != sql_components['limit']:
                validations['limit_correct'] = False
                issues.append(f"Limit mismatch: SQL has {sql_components['limit']}, NoSQL has {nosql_components['limit']}")
        
        # Validate aggregations
        if sql_components['aggregations']:
            if not nosql_components['aggregation_pipeline'] and 'count' not in nosql.lower():
                validations['handles_aggregation'] = False
                issues.append("SQL has aggregation but NoSQL doesn't properly handle it")
        
        # Validate GROUP BY
        if sql_components['group_by']:
            if not nosql_components['has_group']:
                validations['handles_group_by'] = False
                issues.append("SQL has GROUP BY but NoSQL doesn't use $group")
        
        # Validate JOINs
        if sql_components['join_tables']:
            if not ('$lookup' in nosql.lower() or 'join' in nosql.lower()):
                validations['handles_joins'] = False
                issues.append("SQL has JOINs but NoSQL doesn't use $lookup")
        
        # Determine validation level
        validation_count = sum(validations.values())
        total_validations = len(validations)
        validation_percentage = validation_count / total_validations
        
        if validation_percentage == 1.0:
            level = ValidationLevel.CORRECT
        elif validation_percentage >= 0.85:
            level = ValidationLevel.MOSTLY_CORRECT
        elif validation_percentage >= 0.65:
            level = ValidationLevel.MINOR_ISSUES
        elif validation_percentage >= 0.40:
            level = ValidationLevel.MAJOR_ISSUES
        else:
            level = ValidationLevel.INCORRECT
        
        result = {
            'index': index,
            'sql': sql,
            'nosql': nosql,
            'validation_level': level.value,
            'validation_percentage': validation_percentage,
            'validations': validations,
            'issues': issues,
            'sql_components': sql_components,
            'nosql_components': nosql_components,
        }
        
        self.results.append(result)
        return result
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not self.results:
            return {}
        
        total = len(self.results)
        
        level_counts = {
            ValidationLevel.CORRECT.value: 0,
            ValidationLevel.MOSTLY_CORRECT.value: 0,
            ValidationLevel.MINOR_ISSUES.value: 0,
            ValidationLevel.MAJOR_ISSUES.value: 0,
            ValidationLevel.INCORRECT.value: 0,
        }
        
        for result in self.results:
            level_counts[result['validation_level']] += 1
        
        avg_validation = sum(r['validation_percentage'] for r in self.results) / total
        
        # Validation-wise accuracy
        validation_stats = {}
        if self.results:
            all_validations = list(self.results[0]['validations'].keys())
            for val in all_validations:
                passed = sum(1 for r in self.results if r['validations'][val])
                validation_stats[val] = passed / total
        
        # Common issues
        all_issues = []
        for r in self.results:
            all_issues.extend(r['issues'])
        
        summary = {
            'total_queries': total,
            'level_counts': level_counts,
            'level_percentages': {k: (v / total * 100) for k, v in level_counts.items()},
            'avg_validation_percentage': avg_validation,
            'validation_wise_accuracy': validation_stats,
            'correct_rate': (level_counts[ValidationLevel.CORRECT.value] / total * 100),
            'acceptable_rate': ((level_counts[ValidationLevel.CORRECT.value] + 
                                level_counts[ValidationLevel.MOSTLY_CORRECT.value]) / total * 100),
            'total_issues': len(all_issues),
        }
        
        return summary
    
    def print_detailed_report(self, show_individual: bool = True):
        """Print comprehensive validation report"""
        summary = self.get_summary_statistics()
        
        print("\n" + "=" * 100)
        print("NOSQL SEMANTIC VALIDATION REPORT")
        print("=" * 100)
        
        if not summary:
            print("No results to display.")
            return
        
        print(f"\n📊 OVERALL STATISTICS")
        print("-" * 100)
        print(f"Total Queries Validated: {summary['total_queries']}")
        print(f"Average Validation Score: {summary['avg_validation_percentage']:.2%}")
        print(f"Total Issues Found: {summary['total_issues']}")
        
        print(f"\n📈 VALIDATION LEVEL BREAKDOWN")
        print("-" * 100)
        for level, count in summary['level_counts'].items():
            percentage = summary['level_percentages'][level]
            bar = "█" * int(percentage / 2)
            emoji = "✅" if "CORRECT" in level else "⚠️" if "ISSUES" in level else "❌"
            print(f"{emoji} {level:25s}: {count:3d} ({percentage:5.1f}%) {bar}")
        
        print(f"\n🎯 KEY METRICS")
        print("-" * 100)
        print(f"Correct Translations:                    {summary['correct_rate']:5.1f}%")
        print(f"Acceptable (Correct + Mostly Correct):   {summary['acceptable_rate']:5.1f}%")
        
        print(f"\n🔍 VALIDATION CRITERIA PASS RATES")
        print("-" * 100)
        for validation, accuracy in sorted(summary['validation_wise_accuracy'].items(), 
                                          key=lambda x: x[1], reverse=True):
            bar = "█" * int(accuracy * 50)
            emoji = "✅" if accuracy >= 0.8 else "⚠️" if accuracy >= 0.6 else "❌"
            print(f"{emoji} {validation:30s}: {accuracy:5.1%} {bar}")
        
        if show_individual and self.results:
            print(f"\n📝 SAMPLE VALIDATIONS (First 10)")
            print("-" * 100)
            
            for i, result in enumerate(self.results[:10], 1):
                emoji_map = {
                    ValidationLevel.CORRECT.value: "✅",
                    ValidationLevel.MOSTLY_CORRECT.value: "✅",
                    ValidationLevel.MINOR_ISSUES.value: "⚠️",
                    ValidationLevel.MAJOR_ISSUES.value: "⚠️",
                    ValidationLevel.INCORRECT.value: "❌",
                }
                
                emoji = emoji_map[result['validation_level']]
                print(f"\n{emoji} Query {i}: {result['validation_level']} ({result['validation_percentage']:.0%})")
                print(f"   SQL: {result['sql'][:80]}...")
                print(f"   NoSQL: {result['nosql'][:80]}...")
                
                if result['issues']:
                    print(f"   ❌ Issues ({len(result['issues'])}):")
                    for issue in result['issues'][:3]:
                        print(f"      • {issue}")
        
        print("\n" + "=" * 100)
    
    def export_results(self, filename: str = "nosql_validation_results.json"):
        """Export results to JSON"""
        export_data = {
            'summary': self.get_summary_statistics(),
            'individual_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✅ Results exported to {filename}")
