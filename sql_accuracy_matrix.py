"""
SQL Accuracy Matrix - Comprehensive evaluation of generated SQL queries
Compares generated SQL with gold/reference SQL queries using multiple metrics
"""

import re
import difflib
from typing import Dict, List, Tuple, Any
from enum import Enum
import json


class AccuracyLevel(Enum):
    """Accuracy classification levels"""
    EXACT_MATCH = "EXACT_MATCH"
    SEMANTICALLY_EQUIVALENT = "SEMANTICALLY_EQUIVALENT"
    MINOR_DIFFERENCES = "MINOR_DIFFERENCES"
    MAJOR_DIFFERENCES = "MAJOR_DIFFERENCES"
    INCORRECT = "INCORRECT"


class SQLAccuracyMatrix:
    """
    Comprehensive SQL accuracy evaluation matrix
    """
    
    def __init__(self):
        self.results = []
        
    def normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL query for comparison
        - Remove extra whitespace
        - Convert to lowercase
        - Standardize quotes
        """
        sql = sql.strip()
        # Replace multiple spaces with single space
        sql = re.sub(r'\s+', ' ', sql)
        # Normalize quotes
        sql = sql.replace('"', "'")
        # Remove semicolon at end
        sql = sql.rstrip(';')
        return sql.lower()
    
    def extract_sql_components(self, sql: str) -> Dict[str, Any]:
        """
        Extract key components from SQL query for detailed comparison
        """
        sql_lower = sql.lower()
        components = {
            'has_select': 'select' in sql_lower,
            'has_from': 'from' in sql_lower,
            'has_where': 'where' in sql_lower,
            'has_join': 'join' in sql_lower,
            'has_group_by': 'group by' in sql_lower,
            'has_order_by': 'order by' in sql_lower,
            'has_limit': 'limit' in sql_lower,
            'has_distinct': 'distinct' in sql_lower,
            'has_count': 'count(' in sql_lower,
            'has_sum': 'sum(' in sql_lower,
            'has_avg': 'avg(' in sql_lower,
            'has_max': 'max(' in sql_lower,
            'has_min': 'min(' in sql_lower,
        }
        
        # Extract table names (simplified)
        from_match = re.search(r'from\s+(\w+)', sql_lower)
        components['main_table'] = from_match.group(1) if from_match else None
        
        # Extract columns from SELECT (simplified)
        select_match = re.search(r'select\s+(.*?)\s+from', sql_lower)
        if select_match:
            columns = select_match.group(1).split(',')
            components['select_columns'] = [c.strip() for c in columns]
        else:
            components['select_columns'] = []
        
        # Extract WHERE conditions
        where_match = re.search(r'where\s+(.*?)(?:group by|order by|limit|$)', sql_lower)
        components['where_clause'] = where_match.group(1).strip() if where_match else None
        
        return components
    
    def compare_components(self, gen_comp: Dict, gold_comp: Dict) -> Dict[str, bool]:
        """
        Compare SQL components between generated and gold queries
        """
        comparison = {
            'select_match': gen_comp['has_select'] == gold_comp['has_select'],
            'from_match': gen_comp['has_from'] == gold_comp['has_from'],
            'where_match': gen_comp['has_where'] == gold_comp['has_where'],
            'join_match': gen_comp['has_join'] == gold_comp['has_join'],
            'group_by_match': gen_comp['has_group_by'] == gold_comp['has_group_by'],
            'order_by_match': gen_comp['has_order_by'] == gold_comp['has_order_by'],
            'limit_match': gen_comp['has_limit'] == gold_comp['has_limit'],
            'distinct_match': gen_comp['has_distinct'] == gold_comp['has_distinct'],
            'aggregation_match': all([
                gen_comp['has_count'] == gold_comp['has_count'],
                gen_comp['has_sum'] == gold_comp['has_sum'],
                gen_comp['has_avg'] == gold_comp['has_avg'],
                gen_comp['has_max'] == gold_comp['has_max'],
                gen_comp['has_min'] == gold_comp['has_min'],
            ]),
            'table_match': gen_comp['main_table'] == gold_comp['main_table'],
        }
        return comparison
    
    def calculate_similarity_score(self, generated: str, gold: str) -> float:
        """
        Calculate string similarity score using difflib
        Returns value between 0.0 and 1.0
        """
        gen_norm = self.normalize_sql(generated)
        gold_norm = self.normalize_sql(gold)
        
        return difflib.SequenceMatcher(None, gen_norm, gold_norm).ratio()
    
    def evaluate_query(self, generated: str, gold: str, 
                       question: str = "", db_id: str = "") -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single SQL query pair
        """
        # Normalize queries
        gen_norm = self.normalize_sql(generated)
        gold_norm = self.normalize_sql(gold)
        
        # Calculate similarity
        similarity = self.calculate_similarity_score(generated, gold)
        
        # Extract and compare components
        gen_components = self.extract_sql_components(generated)
        gold_components = self.extract_sql_components(gold)
        component_comparison = self.compare_components(gen_components, gold_components)
        
        # Determine accuracy level
        if gen_norm == gold_norm:
            accuracy_level = AccuracyLevel.EXACT_MATCH
        elif similarity >= 0.95:
            accuracy_level = AccuracyLevel.SEMANTICALLY_EQUIVALENT
        elif similarity >= 0.80:
            accuracy_level = AccuracyLevel.MINOR_DIFFERENCES
        elif similarity >= 0.50:
            accuracy_level = AccuracyLevel.MAJOR_DIFFERENCES
        else:
            accuracy_level = AccuracyLevel.INCORRECT
        
        # Calculate component match percentage
        component_matches = sum(component_comparison.values())
        total_components = len(component_comparison)
        component_accuracy = component_matches / total_components if total_components > 0 else 0.0
        
        result = {
            'question': question,
            'db_id': db_id,
            'generated_sql': generated,
            'gold_sql': gold,
            'generated_normalized': gen_norm,
            'gold_normalized': gold_norm,
            'accuracy_level': accuracy_level.value,
            'similarity_score': similarity,
            'component_accuracy': component_accuracy,
            'component_comparison': component_comparison,
            'generated_components': gen_components,
            'gold_components': gold_components,
        }
        
        self.results.append(result)
        return result
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all evaluated queries
        """
        if not self.results:
            return {}
        
        total = len(self.results)
        
        # Count by accuracy level
        accuracy_counts = {
            AccuracyLevel.EXACT_MATCH.value: 0,
            AccuracyLevel.SEMANTICALLY_EQUIVALENT.value: 0,
            AccuracyLevel.MINOR_DIFFERENCES.value: 0,
            AccuracyLevel.MAJOR_DIFFERENCES.value: 0,
            AccuracyLevel.INCORRECT.value: 0,
        }
        
        for result in self.results:
            accuracy_counts[result['accuracy_level']] += 1
        
        # Calculate average scores
        avg_similarity = sum(r['similarity_score'] for r in self.results) / total
        avg_component_accuracy = sum(r['component_accuracy'] for r in self.results) / total
        
        # Component-wise accuracy
        component_stats = {}
        if self.results:
            all_components = list(self.results[0]['component_comparison'].keys())
            for comp in all_components:
                matches = sum(1 for r in self.results if r['component_comparison'][comp])
                component_stats[comp] = matches / total
        
        summary = {
            'total_queries': total,
            'accuracy_counts': accuracy_counts,
            'accuracy_percentages': {k: (v / total * 100) for k, v in accuracy_counts.items()},
            'avg_similarity_score': avg_similarity,
            'avg_component_accuracy': avg_component_accuracy,
            'component_wise_accuracy': component_stats,
            'strict_accuracy': (accuracy_counts[AccuracyLevel.EXACT_MATCH.value] / total * 100),
            'acceptable_accuracy': ((accuracy_counts[AccuracyLevel.EXACT_MATCH.value] + 
                                    accuracy_counts[AccuracyLevel.SEMANTICALLY_EQUIVALENT.value]) / total * 100),
        }
        
        return summary
    
    def print_detailed_report(self, show_individual: bool = True):
        """
        Print comprehensive accuracy report
        """
        summary = self.get_summary_statistics()
        
        print("\n" + "=" * 100)
        print("SQL ACCURACY MATRIX - COMPREHENSIVE EVALUATION REPORT")
        print("=" * 100)
        
        if not summary:
            print("No results to display.")
            return
        
        # Summary statistics
        print(f"\n📊 OVERALL STATISTICS")
        print("-" * 100)
        print(f"Total Queries Evaluated: {summary['total_queries']}")
        print(f"Average Similarity Score: {summary['avg_similarity_score']:.2%}")
        print(f"Average Component Accuracy: {summary['avg_component_accuracy']:.2%}")
        
        # Accuracy breakdown
        print(f"\n📈 ACCURACY BREAKDOWN")
        print("-" * 100)
        for level, count in summary['accuracy_counts'].items():
            percentage = summary['accuracy_percentages'][level]
            bar = "█" * int(percentage / 2)
            print(f"{level:30s}: {count:3d} ({percentage:5.1f}%) {bar}")
        
        print(f"\n🎯 KEY METRICS")
        print("-" * 100)
        print(f"Strict Accuracy (Exact Match):               {summary['strict_accuracy']:5.1f}%")
        print(f"Acceptable Accuracy (Exact + Semantic):      {summary['acceptable_accuracy']:5.1f}%")
        
        # Component-wise accuracy
        print(f"\n🔍 COMPONENT-WISE ACCURACY")
        print("-" * 100)
        for comp, accuracy in sorted(summary['component_wise_accuracy'].items(), 
                                    key=lambda x: x[1], reverse=True):
            bar = "█" * int(accuracy * 50)
            print(f"{comp:25s}: {accuracy:5.1%} {bar}")
        
        # Individual results
        if show_individual and self.results:
            print(f"\n📝 INDIVIDUAL QUERY EVALUATIONS")
            print("-" * 100)
            
            for i, result in enumerate(self.results, 1):
                status_emoji = {
                    AccuracyLevel.EXACT_MATCH.value: "✅",
                    AccuracyLevel.SEMANTICALLY_EQUIVALENT.value: "✅",
                    AccuracyLevel.MINOR_DIFFERENCES.value: "⚠️",
                    AccuracyLevel.MAJOR_DIFFERENCES.value: "⚠️",
                    AccuracyLevel.INCORRECT.value: "❌",
                }
                
                emoji = status_emoji[result['accuracy_level']]
                print(f"\n{emoji} Query {i}: {result['accuracy_level']}")
                
                if result['db_id']:
                    print(f"   Database: {result['db_id']}")
                if result['question']:
                    print(f"   Question: {result['question'][:80]}...")
                
                print(f"   Similarity: {result['similarity_score']:.1%} | "
                      f"Component Match: {result['component_accuracy']:.1%}")
                
                print(f"   Gold SQL:      {result['gold_sql'][:80]}...")
                print(f"   Generated SQL: {result['generated_sql'][:80]}...")
                
                # Show mismatches
                mismatches = [k for k, v in result['component_comparison'].items() if not v]
                if mismatches:
                    print(f"   ⚠️ Mismatches: {', '.join(mismatches)}")
        
        print("\n" + "=" * 100)
    
    def export_results(self, filename: str = "sql_accuracy_results.json"):
        """
        Export results to JSON file
        """
        export_data = {
            'summary': self.get_summary_statistics(),
            'individual_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✅ Results exported to {filename}")


def example_usage():
    """
    Example usage of SQLAccuracyMatrix
    """
    matrix = SQLAccuracyMatrix()
    
    # Example evaluations
    examples = [
        {
            'generated': "SELECT name, age FROM users WHERE age > 18 ORDER BY name",
            'gold': "SELECT name, age FROM users WHERE age > 18 ORDER BY name",
            'question': "Get all users older than 18",
            'db_id': "user_db"
        },
        {
            'generated': "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            'gold': "SELECT COUNT(*) FROM orders WHERE status = 'complete'",
            'question': "Count completed orders",
            'db_id': "order_db"
        },
        {
            'generated': "SELECT * FROM products",
            'gold': "SELECT product_id, name, price FROM products ORDER BY price DESC",
            'question': "List all products",
            'db_id': "product_db"
        },
    ]
    
    for ex in examples:
        matrix.evaluate_query(ex['generated'], ex['gold'], ex['question'], ex['db_id'])
    
    matrix.print_detailed_report()
    matrix.export_results()


if __name__ == '__main__':
    example_usage()
