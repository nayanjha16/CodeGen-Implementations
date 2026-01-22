"""
Generate formatted tables from SQL and NoSQL evaluation results
"""

import json
import pandas as pd
from tabulate import tabulate


def generate_sql_results_table(json_file='spider_sql_accuracy_results.json'):
    """Generate formatted tables for SQL accuracy results"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    
    print("\n" + "=" * 120)
    print("SQL ACCURACY EVALUATION RESULTS")
    print("=" * 120)
    
    # Table 1: Overall Summary
    print("\n📊 OVERALL SUMMARY")
    print("-" * 120)
    summary_data = [
        ["Total Queries", summary['total_queries']],
        ["Average Similarity Score", f"{summary['avg_similarity_score']:.2%}"],
        ["Average Component Accuracy", f"{summary['avg_component_accuracy']:.2%}"],
        ["Strict Accuracy (Exact Match)", f"{summary['strict_accuracy']:.1f}%"],
        ["Acceptable Accuracy (Exact + Semantic)", f"{summary['acceptable_accuracy']:.1f}%"],
    ]
    print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Table 2: Accuracy Level Distribution
    print("\n📈 ACCURACY LEVEL DISTRIBUTION")
    print("-" * 120)
    accuracy_data = []
    for level, count in summary['accuracy_counts'].items():
        percentage = summary['accuracy_percentages'][level]
        accuracy_data.append([
            level,
            count,
            f"{percentage:.1f}%",
            "█" * int(percentage / 2)
        ])
    print(tabulate(accuracy_data, 
                   headers=["Accuracy Level", "Count", "Percentage", "Visual"], 
                   tablefmt="grid"))
    
    # Table 3: Component-wise Accuracy
    print("\n🔍 COMPONENT-WISE ACCURACY")
    print("-" * 120)
    component_data = []
    for comp, accuracy in summary['component_wise_accuracy'].items():
        component_data.append([
            comp.replace('_', ' ').title(),
            f"{accuracy:.2%}",
            "█" * int(accuracy * 50)
        ])
    # Sort by accuracy descending
    component_data.sort(key=lambda x: float(x[1].strip('%')), reverse=True)
    print(tabulate(component_data, 
                   headers=["SQL Component", "Accuracy", "Visual"], 
                   tablefmt="grid"))
    
    # Table 4: Sample Results (top 10)
    print("\n📝 SAMPLE RESULTS (First 10 Queries)")
    print("-" * 120)
    sample_data = []
    for idx, result in enumerate(data['individual_results'][:10], 1):
        sample_data.append([
            idx,
            result['question'][:40] + "..." if len(result['question']) > 40 else result['question'],
            result['db_id'],
            result['accuracy_level'],
            f"{result['similarity_score']:.2%}",
            f"{result['component_accuracy']:.2%}"
        ])
    print(tabulate(sample_data, 
                   headers=["#", "Question", "Database", "Accuracy Level", "Similarity", "Component"], 
                   tablefmt="grid"))


def generate_nosql_results_table(json_file='nosql_validation_results.json'):
    """Generate formatted tables for NoSQL validation results"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    
    print("\n\n" + "=" * 120)
    print("NOSQL SEMANTIC VALIDATION RESULTS")
    print("=" * 120)
    
    # Table 1: Overall Summary
    print("\n📊 OVERALL SUMMARY")
    print("-" * 120)
    summary_data = [
        ["Total Queries", summary['total_queries']],
        ["Average Validation Score", f"{summary['avg_validation_percentage']:.2%}"],
        ["Correct Rate", f"{summary['correct_rate']:.1f}%"],
        ["Acceptable Rate (Correct + Mostly Correct)", f"{summary['acceptable_rate']:.1f}%"],
        ["Total Issues Found", summary['total_issues']],
    ]
    print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Table 2: Validation Level Distribution
    print("\n📈 VALIDATION LEVEL DISTRIBUTION")
    print("-" * 120)
    validation_data = []
    for level, count in summary['level_counts'].items():
        percentage = summary['level_percentages'][level]
        validation_data.append([
            level,
            count,
            f"{percentage:.1f}%",
            "█" * int(percentage / 2)
        ])
    print(tabulate(validation_data, 
                   headers=["Validation Level", "Count", "Percentage", "Visual"], 
                   tablefmt="grid"))
    
    # Table 3: Validation Check Accuracy
    print("\n🔍 VALIDATION CHECK-WISE ACCURACY")
    print("-" * 120)
    check_data = []
    for check, accuracy in summary['validation_wise_accuracy'].items():
        check_data.append([
            check.replace('_', ' ').title(),
            f"{accuracy:.2%}",
            "█" * int(accuracy * 50)
        ])
    # Sort by accuracy descending
    check_data.sort(key=lambda x: float(x[1].strip('%')), reverse=True)
    print(tabulate(check_data, 
                   headers=["Validation Check", "Pass Rate", "Visual"], 
                   tablefmt="grid"))
    
    # Table 4: Sample Results (top 10)
    print("\n📝 SAMPLE RESULTS (First 10 Queries)")
    print("-" * 120)
    sample_data = []
    for idx, result in enumerate(data['individual_results'][:10], 1):
        issues = ", ".join(result['issues'][:2]) if result['issues'] else "None"
        if len(issues) > 60:
            issues = issues[:57] + "..."
        sample_data.append([
            idx,
            result['validation_level'],
            f"{result['validation_percentage']:.2%}",
            issues
        ])
    print(tabulate(sample_data, 
                   headers=["#", "Validation Level", "Score", "Issues"], 
                   tablefmt="grid"))


def generate_combined_summary():
    """Generate a combined comparison table"""
    
    print("\n\n" + "=" * 120)
    print("COMBINED EVALUATION SUMMARY")
    print("=" * 120)
    
    try:
        with open('spider_sql_accuracy_results.json', 'r') as f:
            sql_data = json.load(f)
        with open('nosql_validation_results.json', 'r') as f:
            nosql_data = json.load(f)
        
        comparison_data = [
            ["Metric", "SQL Generation", "NoSQL Translation"],
            ["Total Queries Evaluated", sql_data['summary']['total_queries'], nosql_data['summary']['total_queries']],
            ["Average Accuracy/Validation", 
             f"{sql_data['summary']['avg_similarity_score']:.2%}", 
             f"{nosql_data['summary']['avg_validation_percentage']:.2%}"],
            ["Exact/Correct Match Rate", 
             f"{sql_data['summary']['strict_accuracy']:.1f}%", 
             f"{nosql_data['summary']['correct_rate']:.1f}%"],
            ["Acceptable Quality Rate", 
             f"{sql_data['summary']['acceptable_accuracy']:.1f}%", 
             f"{nosql_data['summary']['acceptable_rate']:.1f}%"],
        ]
        
        print(tabulate(comparison_data[1:], headers=comparison_data[0], tablefmt="grid"))
        
    except Exception as e:
        print(f"Could not generate combined summary: {e}")


def export_to_csv():
    """Export results to CSV files for further analysis"""
    
    print("\n" + "=" * 120)
    print("EXPORTING TO CSV FILES")
    print("=" * 120)
    
    # SQL Results CSV
    try:
        with open('spider_sql_accuracy_results.json', 'r') as f:
            sql_data = json.load(f)
        
        # Individual results
        sql_df = pd.DataFrame(sql_data['individual_results'])
        sql_df = sql_df[['question', 'db_id', 'accuracy_level', 'similarity_score', 'component_accuracy']]
        sql_df.to_csv('sql_results.csv', index=False)
        print("✅ Exported SQL results to: sql_results.csv")
        
        # Summary
        sql_summary_df = pd.DataFrame([sql_data['summary']])
        sql_summary_df.to_csv('sql_summary.csv', index=False)
        print("✅ Exported SQL summary to: sql_summary.csv")
        
    except Exception as e:
        print(f"❌ Could not export SQL results: {e}")
    
    # NoSQL Results CSV
    try:
        with open('nosql_validation_results.json', 'r') as f:
            nosql_data = json.load(f)
        
        # Individual results
        nosql_df = pd.DataFrame(nosql_data['individual_results'])
        nosql_df = nosql_df[['index', 'validation_level', 'validation_percentage', 'issues']]
        nosql_df['issues'] = nosql_df['issues'].apply(lambda x: '; '.join(x) if x else '')
        nosql_df.to_csv('nosql_results.csv', index=False)
        print("✅ Exported NoSQL results to: nosql_results.csv")
        
        # Summary
        nosql_summary_df = pd.DataFrame([nosql_data['summary']])
        nosql_summary_df.to_csv('nosql_summary.csv', index=False)
        print("✅ Exported NoSQL summary to: nosql_summary.csv")
        
    except Exception as e:
        print(f"❌ Could not export NoSQL results: {e}")


if __name__ == "__main__":
    print("🎯 GENERATING EVALUATION RESULTS TABLES\n")
    
    # Generate SQL results tables
    generate_sql_results_table()
    
    # Generate NoSQL results tables
    generate_nosql_results_table()
    
    # Generate combined summary
    generate_combined_summary()
    
    # Export to CSV
    export_to_csv()
    
    print("\n" + "=" * 120)
    print("✅ TABLE GENERATION COMPLETE!")
    print("=" * 120)
    print("\n📁 Output files created:")
    print("   - sql_results.csv (individual SQL results)")
    print("   - sql_summary.csv (SQL summary statistics)")
    print("   - nosql_results.csv (individual NoSQL results)")
    print("   - nosql_summary.csv (NoSQL summary statistics)")
    print("\n")
