"""
Quick test of SQL Accuracy Matrix
Run this to see the matrix in action with sample queries
"""

from sql_accuracy_matrix import SQLAccuracyMatrix


def run_quick_test():
    """
    Quick test with sample SQL query comparisons
    """
    matrix = SQLAccuracyMatrix()
    
    # Test cases demonstrating different accuracy levels
    test_cases = [
        # Perfect match
        {
            'name': 'Perfect Match',
            'generated': "SELECT name, age FROM users WHERE age > 18 ORDER BY name",
            'gold': "SELECT name, age FROM users WHERE age > 18 ORDER BY name",
            'question': "Get all users older than 18, ordered by name",
            'db_id': "test_db_1"
        },
        # Minor whitespace difference
        {
            'name': 'Whitespace Difference',
            'generated': "SELECT name,age FROM users WHERE age>18 ORDER BY name",
            'gold': "SELECT name, age FROM users WHERE age > 18 ORDER BY name",
            'question': "Get all users older than 18, ordered by name",
            'db_id': "test_db_2"
        },
        # Case difference
        {
            'name': 'Case Difference',
            'generated': "SELECT Name, Age FROM Users WHERE Age > 18 ORDER BY Name",
            'gold': "SELECT name, age FROM users WHERE age > 18 ORDER BY name",
            'question': "Get all users older than 18, ordered by name",
            'db_id': "test_db_3"
        },
        # Missing ORDER BY
        {
            'name': 'Missing ORDER BY',
            'generated': "SELECT name, age FROM users WHERE age > 18",
            'gold': "SELECT name, age FROM users WHERE age > 18 ORDER BY name",
            'question': "Get all users older than 18, ordered by name",
            'db_id': "test_db_4"
        },
        # Different aggregation
        {
            'name': 'Different Aggregation',
            'generated': "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            'gold': "SELECT SUM(total) FROM orders WHERE status = 'completed'",
            'question': "Get total from completed orders",
            'db_id': "test_db_5"
        },
        # Missing JOIN
        {
            'name': 'Missing JOIN',
            'generated': "SELECT u.name, o.order_id FROM users u, orders o WHERE u.id = o.user_id",
            'gold': "SELECT u.name, o.order_id FROM users u INNER JOIN orders o ON u.id = o.user_id",
            'question': "Get user names with their orders",
            'db_id': "test_db_6"
        },
        # Wrong table
        {
            'name': 'Wrong Table',
            'generated': "SELECT product_name FROM inventory",
            'gold': "SELECT product_name FROM products",
            'question': "Get all product names",
            'db_id': "test_db_7"
        },
        # Complex query with GROUP BY
        {
            'name': 'Complex with GROUP BY',
            'generated': "SELECT country, COUNT(*) as count FROM users GROUP BY country HAVING COUNT(*) > 10 ORDER BY count DESC",
            'gold': "SELECT country, COUNT(*) as count FROM users GROUP BY country HAVING COUNT(*) > 10 ORDER BY count DESC",
            'question': "Countries with more than 10 users, ordered by count",
            'db_id': "test_db_8"
        },
        # Missing DISTINCT
        {
            'name': 'Missing DISTINCT',
            'generated': "SELECT city FROM customers",
            'gold': "SELECT DISTINCT city FROM customers",
            'question': "Get all unique cities",
            'db_id': "test_db_9"
        },
        # Different WHERE condition
        {
            'name': 'Different WHERE Condition',
            'generated': "SELECT * FROM products WHERE price > 100",
            'gold': "SELECT * FROM products WHERE price >= 100",
            'question': "Get products priced 100 or more",
            'db_id': "test_db_10"
        },
    ]
    
    print("\n" + "="*100)
    print("🧪 QUICK TEST - SQL ACCURACY MATRIX")
    print("="*100)
    print(f"Testing {len(test_cases)} SQL query comparisons...\n")
    
    # Evaluate all test cases
    for i, test in enumerate(test_cases, 1):
        print(f"{i:2d}. Testing: {test['name']:30s}... ", end="", flush=True)
        result = matrix.evaluate_query(
            generated=test['generated'],
            gold=test['gold'],
            question=test['question'],
            db_id=test['db_id']
        )
        
        # Quick status
        if result['accuracy_level'] == 'EXACT_MATCH':
            print("✅ EXACT")
        elif result['accuracy_level'] == 'SEMANTICALLY_EQUIVALENT':
            print("✅ SEMANTIC")
        elif result['accuracy_level'] == 'MINOR_DIFFERENCES':
            print(f"⚠️  MINOR ({result['similarity_score']:.1%})")
        elif result['accuracy_level'] == 'MAJOR_DIFFERENCES':
            print(f"⚠️  MAJOR ({result['similarity_score']:.1%})")
        else:
            print(f"❌ INCORRECT ({result['similarity_score']:.1%})")
    
    # Print full report
    print("\n")
    matrix.print_detailed_report(show_individual=True)
    
    # Export results
    matrix.export_results('test_sql_accuracy_results.json')
    
    return matrix


if __name__ == '__main__':
    run_quick_test()
