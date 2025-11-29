
def load_mini_dataset():
    return [
        {"question":"How many employees are older than 30?",
         "schema":"employees(id, name, age, salary)",
         "answer_sql":"SELECT COUNT(*) FROM employees WHERE age > 30;"},
        {"question":"List the names of employees with salary above 50000.",
         "schema":"employees(id, name, age, salary)",
         "answer_sql":"SELECT name FROM employees WHERE salary > 50000;"},
        {"question":"What is the average salary of all employees?",
         "schema":"employees(id, name, age, salary)",
         "answer_sql":"SELECT AVG(salary) FROM employees;"}
    ]
