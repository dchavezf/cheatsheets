**Python Cheat Sheet for Data Engineers**

### **1. Data Processing Libraries:**

- **Pandas - Data Manipulation:**
  ```python
  import pandas as pd

  # Read CSV file
  df = pd.read_csv('file.csv')

  # Data filtering
  filtered_data = df[df['column'] > 100]

  # Data transformation
  df['new_column'] = df['column1'] + df['column2']

  # Data aggregation
  grouped_data = df.groupby('category').agg({'value': 'sum'})
  ```

- **NumPy - Numerical Computing:**
  ```python
  import numpy as np

  # Array creation
  arr = np.array([1, 2, 3])

  # Mathematical operations
  result = np.mean(arr)
  ```

### **2. Data Storage Libraries:**

- **SQLAlchemy - Database Interaction:**
  ```python
  from sqlalchemy import create_engine

  # Create a database engine
  engine = create_engine('database_connection_string')

  # Execute SQL query
  result = engine.execute('SELECT * FROM table')
  ```

- **Psycopg2 - PostgreSQL Adapter:**
  ```python
  import psycopg2

  # Connect to PostgreSQL
  conn = psycopg2.connect("dbname=test user=postgres")

  # Execute SQL query
  cursor = conn.cursor()
  cursor.execute("SELECT * FROM table")
  ```

- **PySpark - Apache Spark in Python:**
  ```python
  from pyspark.sql import SparkSession

  # Create a Spark session
  spark = SparkSession.builder.appName("example").getOrCreate()

  # Read data from CSV
  df = spark.read.csv('file.csv', header=True)
  ```

### **3. ETL Frameworks:**

- **Apache Airflow - ETL Orchestration:**
  ```python
  from airflow import DAG
  from airflow.operators.python_operator import PythonOperator
  from datetime import datetime

  # Define DAG
  dag = DAG('my_dag', start_date=datetime(2022, 1, 1))

  # Define Python task
  def my_python_function():
      # Your ETL logic here

  python_task = PythonOperator(
      task_id='python_task',
      python_callable=my_python_function,
      dag=dag
  )
  ```

- **Bonobo - Lightweight ETL:**
  ```python
  import bonobo

  # Define ETL pipeline
  def my_etl_graph():
      # Your ETL logic here

  # Execute ETL
  bonobo.run(my_etl_graph)
  ```

### **4. Data Serialization:**

- **JSON:**
  ```python
  import json

  # Serialize Python object to JSON
  json_data = json.dumps({'key': 'value'})

  # Deserialize JSON to Python object
  python_object = json.loads(json_data)
  ```

- **CSV:**
  ```python
  import csv

  # Write to CSV file
  with open('output.csv', 'w', newline='') as csvfile:
      csv_writer = csv.writer(csvfile)
      csv_writer.writerow(['column1', 'column2'])
      csv_writer.writerow(['value1', 'value2'])
  ```

### **5. Data Visualization:**

- **Matplotlib - Plotting Library:**
  ```python
  import matplotlib.pyplot as plt

  # Plotting
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.xlabel('X-axis label')
  plt.ylabel('Y-axis label')
  plt.title('Title')
  plt.show()
  ```

- **Seaborn - Statistical Data Visualization:**
  ```python
  import seaborn as sns

  # Create a heatmap
  sns.heatmap(data=df.corr(), annot=True)
  ```

### **6. Miscellaneous:**

- **Requests - HTTP Library:**
  ```python
  import requests

  # Make an HTTP request
  response = requests.get('https://api.example.com/data')
  data = response.json()
  ```

- **Beautiful Soup - Web Scraping:**
  ```python
  from bs4 import BeautifulSoup

  # Parse HTML content
  soup = BeautifulSoup(html_content, 'html.parser')

  # Extract information from HTML
  title = soup.title.text
  ```

This Python cheat sheet for data engineers covers key libraries and frameworks for data processing, storage, ETL, serialization, visualization, and miscellaneous tasks. Adjust the code snippets based on your specific use cases and requirements.
