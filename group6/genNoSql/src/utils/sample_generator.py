"""
Sample Data Generator
Generates synthetic MongoDB data and queries for testing
"""
from typing import List, Dict, Any
import json
import random
import logging

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generate sample MongoDB databases and queries for testing"""
    
    def __init__(self):
        """Initialize sample data generator"""
        self.samples = []
    
    def generate_ecommerce_database(self) -> Dict[str, Any]:
        """
        Generate an e-commerce database with products, customers, and orders
        
        Returns:
            Dict with 'databases', 'queries', and 'questions'
        """
        # Define database schema
        databases = {
            "Products": {
                "collection": "Products",
                "fields": ["_id", "name", "category", "price", "stock", "reviews"],
                "documents": [
                    {
                        "_id": 1,
                        "name": "Laptop",
                        "category": "Electronics",
                        "price": 999.99,
                        "stock": 50,
                        "reviews": [
                            {"user": "Alice", "rating": 5, "comment": "Great laptop!"},
                            {"user": "Bob", "rating": 4, "comment": "Good value"}
                        ]
                    },
                    {
                        "_id": 2,
                        "name": "Mouse",
                        "category": "Electronics",
                        "price": 29.99,
                        "stock": 200,
                        "reviews": [
                            {"user": "Charlie", "rating": 5, "comment": "Perfect!"}
                        ]
                    },
                    {
                        "_id": 3,
                        "name": "Desk",
                        "category": "Furniture",
                        "price": 299.99,
                        "stock": 20,
                        "reviews": []
                    }
                ]
            },
            "Customers": {
                "collection": "Customers",
                "fields": ["_id", "name", "email", "city", "orders"],
                "documents": [
                    {
                        "_id": 1,
                        "name": "Alice Johnson",
                        "email": "alice@email.com",
                        "city": "New York",
                        "orders": [
                            {"order_id": 101, "product_id": 1, "quantity": 1, "total": 999.99},
                            {"order_id": 102, "product_id": 2, "quantity": 2, "total": 59.98}
                        ]
                    },
                    {
                        "_id": 2,
                        "name": "Bob Smith",
                        "email": "bob@email.com",
                        "city": "Los Angeles",
                        "orders": [
                            {"order_id": 103, "product_id": 3, "quantity": 1, "total": 299.99}
                        ]
                    },
                    {
                        "_id": 3,
                        "name": "Charlie Brown",
                        "email": "charlie@email.com",
                        "city": "Chicago",
                        "orders": []
                    }
                ]
            }
        }
        
        # Define sample queries
        queries = [
            {
                "question": "Find all products in the Electronics category",
                "query": 'db.Products.find({"category": "Electronics"}, {"_id": 0, "name": 1, "price": 1});',
                "schema": "Collection: Products\nFields: category, name, price",
                "predicted_fields": "category, name, price",
                "collection": "Products"
            },
            {
                "question": "Count the total number of products",
                "query": "db.Products.countDocuments({});",
                "schema": "Collection: Products\nFields: (all)",
                "predicted_fields": "(all)",
                "collection": "Products"
            },
            {
                "question": "Find customers who live in New York",
                "query": 'db.Customers.find({"city": "New York"}, {"name": 1, "email": 1, "_id": 0});',
                "schema": "Collection: Customers\nFields: city, name, email",
                "predicted_fields": "city, name, email",
                "collection": "Customers"
            },
            {
                "question": "Get the average price of all products",
                "query": 'db.Products.aggregate([{"$group": {"_id": null, "avgPrice": {"$avg": "$price"}}}, {"$project": {"_id": 0, "avgPrice": 1}}]);',
                "schema": "Collection: Products\nFields: price",
                "predicted_fields": "price",
                "collection": "Products"
            },
            {
                "question": "Find products with price less than $100",
                "query": 'db.Products.find({"price": {"$lt": 100}}, {"name": 1, "price": 1, "_id": 0});',
                "schema": "Collection: Products\nFields: price, name",
                "predicted_fields": "price, name",
                "collection": "Products"
            },
            {
                "question": "Count total number of reviews across all products",
                "query": 'db.Products.aggregate([{"$unwind": "$reviews"}, {"$group": {"_id": null, "count": {"$sum": 1}}}, {"$project": {"_id": 0, "count": 1}}]);',
                "schema": "Collection: Products\nFields: reviews",
                "predicted_fields": "reviews",
                "collection": "Products"
            },
            {
                "question": "Get all product categories",
                "query": 'db.Products.distinct("category");',
                "schema": "Collection: Products\nFields: category",
                "predicted_fields": "category",
                "collection": "Products"
            },
            {
                "question": "Find customers with email containing 'alice'",
                "query": 'db.Customers.find({"email": {"$regex": "alice"}}, {"name": 1, "email": 1, "_id": 0});',
                "schema": "Collection: Customers\nFields: email, name",
                "predicted_fields": "email, name",
                "collection": "Customers"
            },
            {
                "question": "Count how many customers are in each city",
                "query": 'db.Customers.aggregate([{"$group": {"_id": "$city", "count": {"$sum": 1}}}, {"$project": {"city": "$_id", "count": 1, "_id": 0}}]);',
                "schema": "Collection: Customers\nFields: city",
                "predicted_fields": "city",
                "collection": "Customers"
            },
            {
                "question": "Find the most expensive product",
                "query": 'db.Products.find({}, {"name": 1, "price": 1, "_id": 0}).sort({"price": -1}).limit(1);',
                "schema": "Collection: Products\nFields: name, price",
                "predicted_fields": "name, price",
                "collection": "Products"
            }
        ]
        
        return {
            "databases": databases,
            "queries": queries,
            "schema_summary": {
                "Products": ["_id", "name", "category", "price", "stock", "reviews"],
                "Customers": ["_id", "name", "email", "city", "orders"]
            }
        }
    
    def generate_school_database(self) -> Dict[str, Any]:
        """
        Generate a school database with students, courses, and enrollments
        
        Returns:
            Dict with 'databases', 'queries', and 'questions'
        """
        databases = {
            "Students": {
                "collection": "Students",
                "fields": ["_id", "name", "age", "grade", "courses"],
                "documents": [
                    {
                        "_id": 1,
                        "name": "Emma Wilson",
                        "age": 20,
                        "grade": "Sophomore",
                        "courses": [
                            {"course_id": 101, "course_name": "Mathematics", "credits": 3},
                            {"course_id": 102, "course_name": "Physics", "credits": 4}
                        ]
                    },
                    {
                        "_id": 2,
                        "name": "Liam Davis",
                        "age": 19,
                        "grade": "Freshman",
                        "courses": [
                            {"course_id": 103, "course_name": "Chemistry", "credits": 4}
                        ]
                    },
                    {
                        "_id": 3,
                        "name": "Olivia Martinez",
                        "age": 21,
                        "grade": "Junior",
                        "courses": [
                            {"course_id": 101, "course_name": "Mathematics", "credits": 3},
                            {"course_id": 104, "course_name": "Biology", "credits": 4}
                        ]
                    }
                ]
            }
        }
        
        queries = [
            {
                "question": "Find all students who are Freshmen",
                "query": 'db.Students.find({"grade": "Freshman"}, {"name": 1, "age": 1, "_id": 0});',
                "schema": "Collection: Students\nFields: grade, name, age",
                "predicted_fields": "grade, name, age",
                "collection": "Students"
            },
            {
                "question": "Count total number of students",
                "query": "db.Students.countDocuments({});",
                "schema": "Collection: Students\nFields: (all)",
                "predicted_fields": "(all)",
                "collection": "Students"
            },
            {
                "question": "Find students older than 20",
                "query": 'db.Students.find({"age": {"$gt": 20}}, {"name": 1, "age": 1, "_id": 0});',
                "schema": "Collection: Students\nFields: age, name",
                "predicted_fields": "age, name",
                "collection": "Students"
            },
            {
                "question": "Get the average age of all students",
                "query": 'db.Students.aggregate([{"$group": {"_id": null, "avgAge": {"$avg": "$age"}}}, {"$project": {"_id": 0, "avgAge": 1}}]);',
                "schema": "Collection: Students\nFields: age",
                "predicted_fields": "age",
                "collection": "Students"
            },
            {
                "question": "Count total courses enrolled across all students",
                "query": 'db.Students.aggregate([{"$unwind": "$courses"}, {"$group": {"_id": null, "count": {"$sum": 1}}}, {"$project": {"_id": 0, "count": 1}}]);',
                "schema": "Collection: Students\nFields: courses",
                "predicted_fields": "courses",
                "collection": "Students"
            }
        ]
        
        return {
            "databases": databases,
            "queries": queries,
            "schema_summary": {
                "Students": ["_id", "name", "age", "grade", "courses"]
            }
        }
    
    def save_sample_data(self, output_dir: str = "data/samples"):
        """
        Generate and save all sample datasets
        
        Args:
            output_dir: Output directory for sample data
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate datasets
        ecommerce = self.generate_ecommerce_database()
        school = self.generate_school_database()
        
        # Save to JSON files
        with open(f"{output_dir}/ecommerce.json", "w") as f:
            json.dump(ecommerce, f, indent=2)
        
        with open(f"{output_dir}/school.json", "w") as f:
            json.dump(school, f, indent=2)
        
        # Create a combined training examples file
        training_examples = ecommerce["queries"] + school["queries"]
        
        with open(f"{output_dir}/training_examples.json", "w") as f:
            json.dump(training_examples, f, indent=2)
        
        logger.info(f"Sample data saved to {output_dir}")
        logger.info(f"Generated {len(training_examples)} training examples")
        
        # Create README
        readme = """# Sample Data

This directory contains synthetic MongoDB databases and queries for testing the Text-to-NoSQL system.

## Files

- `ecommerce.json`: E-commerce database (Products, Customers) with 10 sample queries
- `school.json`: School database (Students, Courses) with 5 sample queries
- `training_examples.json`: Combined training examples from all datasets

## Usage

To load this data into the system:

```python
from src.utils.sample_generator import SampleDataGenerator

generator = SampleDataGenerator()
generator.save_sample_data("data/samples")
```

To use with MongoDB:

```python
from src.utils.mongo_client import MongoDBClient
import json

# Load sample data
with open("data/samples/ecommerce.json") as f:
    data = json.load(f)

# Create MongoDB collections
client = MongoDBClient()
client.connect()

for coll_name, coll_data in data["databases"].items():
    client.create_collection(coll_name, coll_data["documents"])
```

## Dataset Statistics

- **E-commerce**: 2 collections, 6 documents total, 10 queries
- **School**: 1 collection, 3 documents, 5 queries
- **Total Training Examples**: 15 queries

## Query Types Covered

- Simple find() with filters
- Projections
- Aggregations ($group, $avg, $sum)
- $unwind for nested arrays
- $regex for text matching
- distinct() queries
- countDocuments()
- sort() and limit()
"""
        
        with open(f"{output_dir}/README.md", "w") as f:
            f.write(readme)
        
        return training_examples


def main():
    """Main function to generate sample data"""
    generator = SampleDataGenerator()
    examples = generator.save_sample_data()
    print(f"Generated {len(examples)} training examples")


if __name__ == "__main__":
    main()
