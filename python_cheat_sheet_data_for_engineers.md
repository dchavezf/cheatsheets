**Python Cheat Sheet for Data Engineers**

### **1. Data Processing Libraries:**

- **Pandas - Data Manipulation:**
**Pandas Cheat Sheet for Python Data Analysis**

### **1. Loading Data:**

- **From CSV:**
  ```python
  import pandas as pd

  df = pd.read_csv('file.csv')
  ```

- **From Excel:**
  ```python
  df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
  ```

- **From SQL Database:**
  ```python
  import sqlite3

  conn = sqlite3.connect('database.db')
  df = pd.read_sql_query('SELECT * FROM table', conn)
  ```

### **2. Exploring Data:**

- **Displaying DataFrame:**
  ```python
  print(df)
  ```

- **Data Types and Info:**
  ```python
  df.info()
  ```

- **Descriptive Statistics:**
  ```python
  df.describe()
  ```

### **3. Data Selection and Indexing:**

- **Selecting Columns:**
  ```python
  df['column_name']
  ```

- **Filtering Rows:**
  ```python
  df[df['column_name'] > 50]
  ```

- **Loc and Iloc:**
  ```python
  df.loc[row_indexer, column_indexer]
  df.iloc[row_index, col_index]
  ```

### **4. Data Cleaning:**

- **Handling Missing Values:**
  ```python
  df.dropna()          # Drop rows with NaN
  df.fillna(value)     # Fill NaN with a specific value
  ```

- **Removing Duplicates:**
  ```python
  df.drop_duplicates()
  ```

- **Renaming Columns:**
  ```python
  df.rename(columns={'old_name': 'new_name'}, inplace=True)
  ```

### **5. Data Transformation:**

- **Adding a New Column:**
  ```python
  df['new_column'] = df['column1'] + df['column2']
  ```

- **Applying Functions:**
  ```python
  df['new_column'] = df['column'].apply(lambda x: custom_function(x))
  ```

- **GroupBy and Aggregation:**
  ```python
  df.groupby('category').agg({'value': 'mean'})
  ```

### **6. Data Visualization:**

- **Matplotlib Integration:**
  ```python
  import matplotlib.pyplot as plt

  df.plot(kind='bar', x='category', y='value')
  plt.show()
  ```

- **Seaborn for Statistical Visualization:**
  ```python
  import seaborn as sns

  sns.scatterplot(x='column1', y='column2', data=df)
  ```

### **7. Data Export:**

- **To CSV:**
  ```python
  df.to_csv('output.csv', index=False)
  ```

- **To Excel:**
  ```python
  df.to_excel('output.xlsx', index=False)
  ```

- **To SQL Database:**
  ```python
  conn = sqlite3.connect('new_database.db')
  df.to_sql('new_table', conn, index=False)
  ```

### **8. Time Series Operations:**

- **Converting to DateTime:**
  ```python
  df['date_column'] = pd.to_datetime(df['date_column'])
  ```

- **Resampling and Aggregation:**
  ```python
  df.resample('D').sum()   # Resample to daily frequency
  ```

### **9. Miscellaneous:**

- **Merging DataFrames:**
  ```python
  pd.merge(df1, df2, on='common_column')
  ```

- **String Operations:**
  ```python
  df['text_column'].str.upper()
  ```

- **Handling Categorical Data:**
  ```python
  pd.get_dummies(df, columns=['category_column'])
  ```

This Pandas cheat sheet covers essential operations for data loading, exploration, cleaning, transformation, visualization, and export. Adjust code snippets based on your specific use cases. Refer to the [official Pandas documentation](https://pandas.pydata.org/pandas-docs/stable/) for more detailed information and advanced features.
  ```

- **NumPy - Numerical Computing:**
**NumPy Cheat Sheet for Python Data Analysis**

### **1. Importing NumPy:**

```python
import numpy as np
```

### **2. Creating Arrays:**

- **From Python Lists:**
  ```python
  arr = np.array([1, 2, 3])
  ```

- **Zeros and Ones Arrays:**
  ```python
  zeros_arr = np.zeros((3, 3))
  ones_arr = np.ones((2, 2))
  ```

- **Identity Matrix:**
  ```python
  identity_mat = np.eye(3)
  ```

### **3. Array Operations:**

- **Element-wise Operations:**
  ```python
  result = arr1 + arr2
  result = arr1 * arr2
  ```

- **Matrix Multiplication:**
  ```python
  mat_mult = np.dot(matrix1, matrix2)
  ```

- **Transpose:**
  ```python
  transposed_mat = matrix.T
  ```

### **4. Indexing and Slicing:**

- **Indexing:**
  ```python
  value = arr[1]
  ```

- **Slicing:**
  ```python
  sub_array = arr[1:4]
  ```

- **Boolean Indexing:**
  ```python
  bool_indexed = arr[arr > 2]
  ```

### **5. Array Shape and Reshaping:**

- **Shape of Array:**
  ```python
  shape = arr.shape
  ```

- **Reshape:**
  ```python
  reshaped_arr = arr.reshape((2, 3))
  ```

### **6. Statistical Operations:**

- **Mean, Median, and Standard Deviation:**
  ```python
  mean_val = np.mean(arr)
  median_val = np.median(arr)
  std_dev = np.std(arr)
  ```

### **7. Random Sampling:**

- **Random Integers:**
  ```python
  random_ints = np.random.randint(low, high, size=(3, 3))
  ```

- **Random Normal Distribution:**
  ```python
  random_values = np.random.normal(mean, std_dev, size=(2, 2))
  ```

### **8. Linear Algebra Operations:**

- **Matrix Determinant:**
  ```python
  determinant = np.linalg.det(matrix)
  ```

- **Eigenvalues and Eigenvectors:**
  ```python
  eigenvalues, eigenvectors = np.linalg.eig(matrix)
  ```

### **9. Stacking and Splitting:**

- **Vertical and Horizontal Stack:**
  ```python
  vertical_stack = np.vstack((arr1, arr2))
  horizontal_stack = np.hstack((arr1, arr2))
  ```

- **Splitting:**
  ```python
  sub_arrays = np.split(arr, 3)
  ```

### **10. Broadcasting:**

- **Broadcasting Operations:**
  ```python
  result = arr + 5
  ```

### **11. File I/O:**

- **Saving and Loading Arrays:**
  ```python
  np.save('array.npy', arr)
  loaded_arr = np.load('array.npy')
  ```

### **12. Advanced Functions:**

- **Vectorized Functions:**
  ```python
  sin_values = np.sin(arr)
  ```

- **Universal Functions (ufunc):**
  ```python
  result = np.add(arr1, arr2)
  ```

This NumPy cheat sheet covers fundamental operations for array creation, manipulation, indexing, statistical operations, linear algebra, random sampling, stacking, and more. Adjust code snippets based on your specific use cases. Refer to the [official NumPy documentation](https://numpy.org/doc/stable/) for more detailed information and advanced features.

### **2. Data Storage Libraries:**

- **SQLAlchemy - Database Interaction:**
  **SQLAlchemy Cheat Sheet for Python**

### **1. Installing SQLAlchemy:**

```bash
pip install sqlalchemy
```

### **2. Creating an Engine:**

```python
from sqlalchemy import create_engine

# SQLite in-memory database
engine = create_engine('sqlite:///:memory:')

# SQLite persistent database
engine = create_engine('sqlite:///your_database.db')

# MySQL
engine = create_engine('mysql://user:password@localhost/db_name')

# PostgreSQL
engine = create_engine('postgresql://user:password@localhost/db_name')
```

### **3. Defining and Creating Tables:**

```python
from sqlalchemy import Table, Column, Integer, String, MetaData

metadata = MetaData()

# Define a table
users = Table('users', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('age', Integer)
)

# Create tables in the database
metadata.create_all(engine)
```

### **4. CRUD Operations:**

- **Inserting Data:**

```python
from sqlalchemy import insert

# Insert single row
stmt = insert(users).values(name='John', age=25)
result = engine.execute(stmt)

# Insert multiple rows
stmt = insert(users).values([
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 22}
])
result = engine.execute(stmt)
```

- **Querying Data:**

```python
from sqlalchemy import select

# Select all
stmt = select(users)
result = engine.execute(stmt)

# Fetch all results
rows = result.fetchall()

# Select with conditions
stmt = select(users).where(users.c.age > 25)
result = engine.execute(stmt)
```

- **Updating Data:**

```python
from sqlalchemy import update

# Update single row
stmt = update(users).where(users.c.name == 'John').values(age=26)
result = engine.execute(stmt)
```

- **Deleting Data:**

```python
from sqlalchemy import delete

# Delete data
stmt = delete(users).where(users.c.age < 25)
result = engine.execute(stmt)
```

### **5. Relationships:**

```python
from sqlalchemy import ForeignKey

addresses = Table('addresses', metadata,
    Column('id', Integer, primary_key=True),
    Column('email', String),
    Column('user_id', Integer, ForeignKey('users.id'))
)

# Create foreign key relationship
users.append_column(Column('id', Integer, primary_key=True))
metadata.create_all(engine)
```

### **6. Using ORM (Object-Relational Mapping):**

```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)
    addresses = relationship('Address', back_populates='user')

class Address(Base):
    __tablename__ = 'addresses'
    id = Column(Integer, primary_key=True)
    email = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship('User', back_populates='addresses')

# Create tables
Base.metadata.create_all(engine)
```

### **7. Session and Transactions:**

```python
from sqlalchemy.orm import sessionmaker

# Create a Session class
Session = sessionmaker(bind=engine)

# Create a Session instance
session = Session()

# Begin a transaction
with session.begin():
    # CRUD operations using session
    user = User(name='John', age=25)
    session.add(user)
    session.commit()
```

### **8. Querying with ORM:**

```python
# Querying data
john = session.query(User).filter_by(name='John').first()
```

### **9. Connection Pooling:**

```python
from sqlalchemy.pool import QueuePool

# Use QueuePool for SQLite
engine = create_engine('sqlite:///:memory:', poolclass=QueuePool)
```

This SQLAlchemy cheat sheet covers essential operations for database connectivity, table creation, CRUD operations, relationships, ORM usage, and more. Adjust code snippets based on your specific database and use cases. Refer to the [official SQLAlchemy documentation](https://docs.sqlalchemy.org/en/20/) for more detailed information and advanced features.
- **Psycopg2 - PostgreSQL Adapter:**
**Psycopg2 Cheat Sheet for Python**

### **1. Installing Psycopg2:**

```bash
pip install psycopg2
```

### **2. Connecting to PostgreSQL:**

```python
import psycopg2

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host="your_host",
    database="your_database",
    user="your_user",
    password="your_password",
    port="your_port"
)

# Create a cursor
cur = conn.cursor()
```

### **3. Executing SQL Queries:**

- **Executing a Query:**

```python
cur.execute("SELECT * FROM your_table")
```

- **Executing a Query with Parameters:**

```python
cur.execute("SELECT * FROM your_table WHERE column_name = %s", (value,))
```

### **4. Fetching Results:**

- **Fetching a Single Row:**

```python
row = cur.fetchone()
```

- **Fetching Multiple Rows:**

```python
rows = cur.fetchmany(5)  # Fetch 5 rows
```

- **Fetching All Rows:**

```python
all_rows = cur.fetchall()
```

### **5. Modifying Data:**

- **Inserting Data:**

```python
cur.execute("INSERT INTO your_table (column1, column2) VALUES (%s, %s)", (value1, value2))
```

- **Updating Data:**

```python
cur.execute("UPDATE your_table SET column_name = %s WHERE condition_column = %s", (new_value, condition_value))
```

- **Deleting Data:**

```python
cur.execute("DELETE FROM your_table WHERE column_name = %s", (value,))
```

### **6. Transactions:**

- **Committing Changes:**

```python
conn.commit()
```

- **Rolling Back Changes:**

```python
conn.rollback()
```

### **7. Handling Exceptions:**

```python
try:
    # Database operations
except psycopg2.Error as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    # Close the cursor and connection
    cur.close()
    conn.close()
```

### **8. Connection Pooling:**

```python
from psycopg2 import pool

# Create a connection pool
connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host="your_host",
    database="your_database",
    user="your_user",
    password="your_password",
    port="your_port"
)

# Get a connection from the pool
conn = connection_pool.getconn()

# Perform database operations

# Release the connection back to the pool
connection_pool.putconn(conn)
```

### **9. Server-Side Cursors:**

```python
cur = conn.cursor(name='server_side_cursor')
cur.execute("SELECT * FROM your_table")
```

### **10. Using DictCursor:**

```python
from psycopg2.extras import DictCursor

cur = conn.cursor(cursor_factory=DictCursor)
cur.execute("SELECT * FROM your_table")
row = cur.fetchone()
```

This Psycopg2 cheat sheet covers fundamental operations for connecting to PostgreSQL, executing queries, fetching results, modifying data, handling transactions, and more. Adjust code snippets based on your specific PostgreSQL setup and use cases. Refer to the [official Psycopg2 documentation](https://www.psycopg.org/docs/) for more detailed information and advanced features.

- **PySpark - Apache Spark in Python:**
**Apache Spark Cheat Sheet for Python (PySpark)**

### **1. Installing PySpark:**

```bash
pip install pyspark
```

### **2. Initializing Spark Session:**

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("my_app").getOrCreate()
```

### **3. Loading Data:**

- **From CSV:**

```python
df = spark.read.csv('file.csv', header=True, inferSchema=True)
```

- **From Parquet:**

```python
df = spark.read.parquet('file.parquet')
```

### **4. Data Exploration:**

- **Displaying DataFrame:**

```python
df.show()
```

- **Data Types and Schema:**

```python
df.printSchema()
```

### **5. Transformations:**

- **Selecting Columns:**

```python
df.select('column1', 'column2')
```

- **Filtering Rows:**

```python
df.filter(df['column'] > 50)
```

- **GroupBy and Aggregation:**

```python
df.groupBy('category').agg({'value': 'mean'})
```

### **6. DataFrame Operations:**

- **Joining DataFrames:**

```python
joined_df = df1.join(df2, on='common_column', how='inner')
```

- **Union:**

```python
union_df = df1.union(df2)
```

- **Pivot:**

```python
pivot_df = df.groupBy('category').pivot('date').sum('value')
```

### **7. SQL Queries:**

```python
df.createOrReplaceTempView('my_table')

result = spark.sql("SELECT * FROM my_table WHERE column > 50")
```

### **8. Machine Learning with MLlib:**

- **Linear Regression:**

```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['feature1', 'feature2'], outputCol='features')
df = assembler.transform(df)

lr = LinearRegression(featuresCol='features', labelCol='label')
model = lr.fit(df)
```

### **9. Saving Data:**

- **Save to Parquet:**

```python
df.write.parquet('output.parquet')
```

- **Save to CSV:**

```python
df.write.csv('output.csv', header=True)
```

### **10. Spark Configurations:**

- **Setting Configurations:**

```python
spark.conf.set("spark.executor.memory", "2g")
```

### **11. Handling Missing Data:**

```python
df.na.drop()  # Drop rows with missing values
```

### **12. RDD Operations:**

```python
rdd = df.rdd

# Map operation
result_rdd = rdd.map(lambda x: x['column'] * 2)

# Filter operation
filtered_rdd = rdd.filter(lambda x: x['column'] > 50)
```

### **13. Spark Streaming:**

```python
from pyspark.streaming import StreamingContext

# Create a StreamingContext
ssc = StreamingContext(spark, batchDuration=10)

# Define a DStream from a data source
dstream = ssc.textFileStream('input_directory')

# Perform streaming operations
result_stream = dstream.flatMap(lambda line: line.split(' ')).countByValue()
```

### **14. GraphX:**

- **Creating a Graph:**

```python
from pyspark.graphx import Graph

vertices = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])
edges = spark.createDataFrame([(1, 2, "friend"), (2, 1, "friend")], ["src", "dst", "relationship"])

graph = Graph(vertices, edges)
```

This PySpark cheat sheet covers essential operations for initializing Spark, loading data, data exploration, transformations, DataFrame operations, machine learning with MLlib, saving data, Spark configurations, handling missing data, RDD operations, Spark Streaming, and GraphX. Adjust code snippets based on your specific use cases. Refer to the [official PySpark documentation](https://spark.apache.org/docs/latest/api/python/index.html) for more detailed information and advanced features.

### **3. ETL Frameworks:**

- **Apache Airflow - ETL Orchestration:**
 **Apache Airflow Cheat Sheet for Python**

### **1. Installing Apache Airflow:**

```bash
pip install apache-airflow
```

### **2. Initializing Airflow Database:**

```bash
airflow db init
```

### **3. Starting the Airflow Web Server:**

```bash
airflow webserver
```

### **4. Starting the Airflow Scheduler:**

```bash
airflow scheduler
```

### **5. Configuring Airflow:**

- **Airflow Configuration File:**
  Default location: `~/airflow/airflow.cfg`

- **Setting Variables:**
  Use the Airflow UI or CLI to set variables.

### **6. Creating a DAG (Directed Acyclic Graph):**

```python
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator

default_args = {
    'owner': 'your_name',
    'start_date': datetime(2022, 1, 1),
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'your_dag_id',
    default_args=default_args,
    description='Your DAG description',
    schedule_interval=timedelta(days=1),
)
```

### **7. Defining Tasks in a DAG:**

```python
def task_function(**kwargs):
    # Your task logic

task1 = PythonOperator(
    task_id='task_id1',
    python_callable=task_function,
    provide_context=True,
    dag=dag,
)
```

### **8. Setting Task Dependencies:**

```python
task1 >> task2  # task2 depends on task1
```

### **9. Operators:**

- **BashOperator:**

```python
from airflow.operators.bash_operator import BashOperator

task = BashOperator(
    task_id='task_id',
    bash_command='your_bash_command',
    dag=dag,
)
```

- **PythonOperator:**

```python
from airflow.operators.python_operator import PythonOperator

task = PythonOperator(
    task_id='task_id',
    python_callable=your_python_function,
    provide_context=True,
    dag=dag,
)
```

- **DummyOperator (Placeholder):**

```python
from airflow.operators.dummy_operator import DummyOperator

dummy_task = DummyOperator(
    task_id='dummy_task',
    dag=dag,
)
```

### **10. Sensors:**

- **ExternalTaskSensor:**

```python
from airflow.sensors.external_task_sensor import ExternalTaskSensor

sensor_task = ExternalTaskSensor(
    task_id='sensor_task',
    external_dag_id='external_dag_id',
    external_task_id='external_task_id',
    dag=dag,
)
```

### **11. Trigger Rules:**

- **Setting Trigger Rule:**

```python
task1 >> task2  # task2 runs regardless of success/failure of task1
```

### **12. Templating:**

- **Using Jinja Templating:**

```python
task = PythonOperator(
    task_id='task_id',
    python_callable=your_python_function,
    op_args=['{{ var.value }}'],
    provide_context=True,
    dag=dag,
)
```

### **13. Hooks:**

- **Using Hooks:**

```python
from airflow.hooks.base_hook import BaseHook

conn = BaseHook.get_connection('your_connection_id')
```

### **14. XCom:**

- **Pushing and Pulling XComs:**

```python
# Pushing XCom from task
context['ti'].xcom_push(key='key_name', value='value')

# Pulling XCom in another task
value = context['ti'].xcom_pull(task_ids='previous_task_id', key='key_name')
```

### **15. Airflow CLI:**

- **Running a DAG:**

```bash
airflow dags trigger -e 2022-01-01 your_dag_id
```

- **Testing a Task:**

```bash
airflow tasks test your_dag_id task_id1 2022-01-01
```

### **16. Airflow UI:**

- **Accessing the UI:**
  Open a web browser and navigate to `http://localhost:8080`

### **17. Advanced Features:**

- **Variables:**
  Store and manage variables in the Airflow UI.

- **Connections:**
  Manage external system connections in the Airflow UI.

- **Templates:**
  Use templates for dynamic values in DAGs.

This Apache Airflow cheat sheet covers fundamental operations for installing, configuring, creating DAGs, defining tasks, setting dependencies, using operators and sensors, configuring triggers, templating, using hooks and XCom, and accessing the Airflow CLI and UI. Adjust code snippets based on your specific use cases. Refer to the [official Apache Airflow documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html) for more detailed information and advanced features.

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
  **JSON (JavaScript Object Notation) Serialization Cheat Sheet in Python**

### **1. Importing JSON Module:**

```python
import json
```

### **2. Serializing Python Objects to JSON:**

- **Serialize a Dictionary:**

```python
data = {'name': 'John', 'age': 30, 'city': 'New York'}
json_data = json.dumps(data, indent=2)
```

- **Serialize a List:**

```python
data_list = [1, 'apple', True, {'color': 'red'}]
json_data_list = json.dumps(data_list, indent=2)
```

### **3. Writing JSON to a File:**

```python
with open('output.json', 'w') as file:
    json.dump(data, file, indent=2)
```

### **4. Deserializing JSON to Python Objects:**

- **Deserialize to Dictionary:**

```python
json_data = '{"name": "John", "age": 30, "city": "New York"}'
python_dict = json.loads(json_data)
```

- **Deserialize to List:**

```python
json_data_list = '[1, "apple", true, {"color": "red"}]'
python_list = json.loads(json_data_list)
```

### **5. Reading JSON from a File:**

```python
with open('input.json', 'r') as file:
    data_from_file = json.load(file)
```

### **6. Handling JSON Encoding and Decoding Errors:**

```python
try:
    json_data = json.dumps(non_serializable_object)
except json.JSONDecodeError as e:
    print(f"Error: {e}")
```

### **7. Advanced Serialization Options:**

- **Custom Serialization Function:**

```python
def custom_serializer(obj):
    if isinstance(obj, datetime.datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError("Type not serializable")

data = {'name': 'John', 'timestamp': datetime.datetime.now()}
json_data = json.dumps(data, default=custom_serializer, indent=2)
```

- **JSON Encoder Class:**

```python
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

data = {'name': 'John', 'timestamp': datetime.datetime.now()}
json_data = json.dumps(data, cls=CustomEncoder, indent=2)
```

### **8. Pretty Printing JSON:**

```python
json_data = '{"name": "John", "age": 30, "city": "New York"}'
parsed_data = json.loads(json_data)
pretty_json = json.dumps(parsed_data, indent=2)
print(pretty_json)
```

### **9. Handling Decimal Serialization:**

```python
from decimal import Decimal

data = {'price': Decimal('19.99')}
json_data = json.dumps(data, indent=2, default=str)
```

### **10. JSON Schema Validation:**

- **Using `jsonschema` Library:**

```bash
pip install jsonschema
```

```python
import jsonschema

schema = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'age': {'type': 'number'},
        'city': {'type': 'string'},
    },
    'required': ['name', 'age']
}

json_data = '{"name": "John", "age": 30, "city": "New York"}'

try:
    jsonschema.validate(json.loads(json_data), schema)
    print("Validation successful!")
except jsonschema.exceptions.ValidationError as e:
    print(f"Validation error: {e}")
```

This JSON serialization cheat sheet covers fundamental concepts and techniques for working with JSON data in Python. Adjust code snippets based on your specific use cases. Refer to the [official Python documentation](https://docs.python.org/3/library/json.html) for more detailed information and advanced features.
  ```

- **CSV:**
**CSV (Comma-Separated Values) Serialization Cheat Sheet in Python**

### **1. Importing CSV Module:**

```python
import csv
```

### **2. Writing to CSV:**

- **Write a Single Row:**

```python
data = ['John', 30, 'New York']

with open('output.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(data)
```

- **Write Multiple Rows:**

```python
data_list = [['John', 30, 'New York'], ['Alice', 25, 'Los Angeles']]

with open('output.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data_list)
```

### **3. Writing to CSV with DictWriter:**

```python
fieldnames = ['Name', 'Age', 'City']
data_list_of_dicts = [
    {'Name': 'John', 'Age': 30, 'City': 'New York'},
    {'Name': 'Alice', 'Age': 25, 'City': 'Los Angeles'}
]

with open('output.csv', 'w', newline='') as file:
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(data_list_of_dicts)
```

### **4. Reading from CSV:**

- **Read the Entire CSV File:**

```python
with open('input.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        print(row)
```

- **Read Specific Rows:**

```python
with open('input.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for i in range(5):  # Read first 5 rows
        row = next(csv_reader)
        print(row)
```

### **5. Reading from CSV with DictReader:**

```python
with open('input.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        print(row)
```

### **6. Handling Delimiters and Quote Characters:**

- **Custom Delimiter:**

```python
with open('input.tsv', 'r') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
        print(row)
```

- **Custom Quote Character:**

```python
with open('input.csv', 'r') as file:
    csv_reader = csv.reader(file, quotechar="'")
    for row in csv_reader:
        print(row)
```

### **7. Writing/Reading CSV to/from a List of Dictionaries:**

```python
data_list_of_dicts = [
    {'Name': 'John', 'Age': 30, 'City': 'New York'},
    {'Name': 'Alice', 'Age': 25, 'City': 'Los Angeles'}
]

# Writing to CSV
with open('output.csv', 'w', newline='') as file:
    fieldnames = data_list_of_dicts[0].keys()
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(data_list_of_dicts)

# Reading from CSV
with open('input.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        print(row)
```

### **8. Handling CSV Errors:**

```python
try:
    with open('input.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            print(row)
except csv.Error as e:
    print(f"CSV Error: {e}")
```

### **9. CSV Sniffer:**

- **Detecting CSV Properties:**

```python
with open('input.csv', 'r') as file:
    sample_data = file.read(1024)  # Read a sample to determine properties
    sniffer = csv.Sniffer()
    has_header = sniffer.has_header(sample_data)
    delimiter = sniffer.sniff(sample_data).delimiter

    print(f"Has Header: {has_header}")
    print(f"Delimiter: {delimiter}")
```

### **10. Handling Unicode Encoding/Decoding:**

```python
with open('input.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        print(row)
```

This CSV serialization cheat sheet covers essential concepts and techniques for working with CSV data in Python. Adjust code snippets based on your specific use cases. Refer to the [official Python documentation](https://docs.python.org/3/library/csv.html) for more detailed information and advanced features.

### **5. Data Visualization:**

- **Matplotlib - Plotting Library:**
  **Matplotlib Cheat Sheet for Python**

### **1. Installing Matplotlib:**

```bash
pip install matplotlib
```

### **2. Importing Matplotlib:**

```python
import matplotlib.pyplot as plt
```

### **3. Basic Plotting:**

- **Line Plot:**

```python
x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 12, 9]

plt.plot(x, y, label='Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.legend()
plt.show()
```

- **Scatter Plot:**

```python
plt.scatter(x, y, label='Scatter Plot', color='red', marker='o')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Scatter Plot')
plt.legend()
plt.show()
```

### **4. Customizing Plots:**

- **Colors and Markers:**

```python
plt.plot(x, y, color='green', linestyle='--', marker='o', markersize=8, label='Custom Plot')
```

- **Adding Grid:**

```python
plt.grid(True)
```

- **Setting Limits:**

```python
plt.xlim(0, 6)
plt.ylim(0, 20)
```

- **Adding Annotations:**

```python
plt.annotate('Annotation', xy=(3, 10), xytext=(3.5, 12), arrowprops=dict(facecolor='black', shrink=0.05))
```

### **5. Subplots:**

```python
plt.subplot(2, 1, 1)
plt.plot(x, y, label='Subplot 1')

plt.subplot(2, 1, 2)
plt.scatter(x, y, label='Subplot 2', color='red', marker='o')

plt.show()
```

### **6. Bar Charts:**

```python
categories = ['Category 1', 'Category 2', 'Category 3']
values = [15, 20, 10]

plt.bar(categories, values, color='blue', alpha=0.7)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()
```

### **7. Histogram:**

```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

plt.hist(data, bins=5, color='purple', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
```

### **8. Pie Chart:**

```python
labels = ['Label 1', 'Label 2', 'Label 3']
sizes = [30, 40, 30]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['gold', 'lightcoral', 'lightskyblue'])
plt.title('Pie Chart')
plt.show()
```

### **9. Saving Plots:**

```python
plt.savefig('plot.png')
```

### **10. Advanced Plotting:**

- **3D Plot:**

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 12, 9]
z = [5, 8, 2, 4, 7]

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()
```

- **Heatmap:**

```python
import numpy as np

data = np.random.random((5, 5))

plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Heatmap')
plt.show()
```

This Matplotlib cheat sheet covers basic to advanced plotting techniques, customization options, and different plot types. Adjust code snippets based on your specific use cases. Refer to the [official Matplotlib documentation](https://matplotlib.org/stable/contents.html) for more detailed information and advanced features.
- **Seaborn - Statistical Data Visualization:**
**Seaborn Cheat Sheet for Python**

### **1. Installing Seaborn:**

```bash
pip install seaborn
```

### **2. Importing Seaborn:**

```python
import seaborn as sns
```

### **3. Setting Style:**

- **Default Style:**

```python
sns.set()
```

- **Available Styles:**

```python
sns.set_style("whitegrid")  # Other options: "darkgrid", "white", "dark", "ticks"
```

### **4. Plotting with Seaborn:**

- **Scatter Plot:**

```python
sns.scatterplot(x='x', y='y', data=df)
```

- **Line Plot:**

```python
sns.lineplot(x='x', y='y', data=df)
```

### **5. Categorical Plots:**

- **Bar Plot:**

```python
sns.barplot(x='category', y='value', data=df)
```

- **Count Plot:**

```python
sns.countplot(x='category', data=df)
```

### **6. Distribution Plots:**

- **Histogram:**

```python
sns.histplot(x='values', data=df, bins=20, kde=True)
```

- **Kernel Density Estimation (KDE) Plot:**

```python
sns.kdeplot(x='values', data=df)
```

### **7. Box and Violin Plots:**

- **Box Plot:**

```python
sns.boxplot(x='category', y='values', data=df)
```

- **Violin Plot:**

```python
sns.violinplot(x='category', y='values', data=df)
```

### **8. Pair Plots:**

```python
sns.pairplot(df, hue='category')
```

### **9. Heatmap:**

```python
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
```

### **10. Styling Plots:**

- **Changing Color Palette:**

```python
sns.set_palette("pastel")  # Other options: "deep", "muted", "bright", "colorblind", etc.
```

- **Customizing Plot Aesthetics:**

```python
sns.set_context("notebook")  # Other options: "paper", "talk", "poster", etc.
```

### **11. Regression Plots:**

- **Linear Regression:**

```python
sns.regplot(x='x', y='y', data=df)
```

### **12. Facet Grids:**

```python
g = sns.FacetGrid(df, col='category', hue='category')
g.map(sns.scatterplot, 'x', 'y')
g.add_legend()
```

### **13. Adding Annotations:**

```python
ax = sns.scatterplot(x='x', y='y', data=df)
ax.annotate('Annotation', xy=(3, 10), xytext=(3.5, 12), arrowprops=dict(facecolor='black', shrink=0.05))
```

### **14. Customizing Axes:**

- **Setting Axis Limits:**

```python
plt.xlim(0, 10)
plt.ylim(0, 20)
```

- **Rotating Axis Labels:**

```python
plt.xticks(rotation=45)
```

### **15. Saving Plots:**

```python
plt.savefig('seaborn_plot.png')
```

This Seaborn cheat sheet covers essential plotting techniques, customization options, and different plot types. Adjust code snippets based on your specific use cases. Refer to the [official Seaborn documentation](https://seaborn.pydata.org/) for more detailed information and advanced features.

### **6. Miscellaneous:**

- **Requests - HTTP Library:**
**API Information Retrieval in Python Cheat Sheet**

### **1. Installing Necessary Libraries:**

```bash
pip install requests
```

### **2. Making a GET Request:**

```python
import requests

url = 'https://api.example.com/data'
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")
```

### **3. Adding Query Parameters:**

```python
params = {'param1': 'value1', 'param2': 'value2'}
response = requests.get(url, params=params)
```

### **4. Handling Authentication:**

- **Basic Authentication:**

```python
from requests.auth import HTTPBasicAuth

username = 'your_username'
password = 'your_password'
response = requests.get(url, auth=HTTPBasicAuth(username, password))
```

- **API Key Authentication:**

```python
headers = {'API-Key': 'your_api_key'}
response = requests.get(url, headers=headers)
```

### **5. Handling Headers:**

```python
headers = {'Content-Type': 'application/json', 'User-Agent': 'YourApp/1.0'}
response = requests.get(url, headers=headers)
```

### **6. Handling JSON Response:**

```python
data = response.json()
print(data)
```

### **7. Error Handling:**

```python
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### **8. POST Request:**

```python
import json

url = 'https://api.example.com/create'
payload = {'key1': 'value1', 'key2': 'value2'}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(payload), headers=headers)
```

### **9. PUT Request:**

```python
response = requests.put(url, data=json.dumps(payload), headers=headers)
```

### **10. DELETE Request:**

```python
response = requests.delete(url, headers=headers)
```

### **11. Handling Timeouts:**

```python
timeout = 5  # in seconds
response = requests.get(url, timeout=timeout)
```

### **12. Handling Query Parameters in URL:**

```python
import urllib.parse

url = 'https://api.example.com/data'
query_params = {'param1': 'value1', 'param2': 'value2'}
url_with_params = f"{url}?{urllib.parse.urlencode(query_params)}"
response = requests.get(url_with_params)
```

### **13. Using Session for Persistent Connection:**

```python
session = requests.Session()
response = session.get(url)
```

### **14. Rate Limiting:**

- **Using `time.sleep()`:**

```python
import time

response = requests.get(url)
if response.status_code == 429:  # Too Many Requests
    retry_after = int(response.headers['Retry-After'])
    time.sleep(retry_after)
    response = requests.get(url)
```

This cheat sheet covers essential concepts for making API requests in Python using the `requests` library. Adjust the code snippets based on the specific API you are working with and the details of the endpoints you need to interact with. Refer to the [official requests documentation](https://docs.python-requests.org/en/latest/) for more details and advanced features.

- **Beautiful Soup - Web Scraping:**
**Beautiful Soup - Web Scraping in Python Cheat Sheet**

### **1. Installing Necessary Libraries:**

```bash
pip install beautifulsoup4
```

### **2. Importing Beautiful Soup:**

```python
from bs4 import BeautifulSoup
```

### **3. Making a GET Request:**

```python
import requests

url = 'https://example.com'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
```

### **4. Parsing HTML Content:**

- **HTML String:**

```python
html_string = "<html><body><p>Hello, Beautiful Soup!</p></body></html>"
soup = BeautifulSoup(html_string, 'html.parser')
```

### **5. Navigating the HTML Tree:**

- **Accessing Tags:**

```python
title_tag = soup.title
```

- **Navigating Up and Down:**

```python
parent_tag = title_tag.parent
next_sibling_tag = title_tag.next_sibling
```

### **6. Searching for Tags:**

- **Finding Single Tag:**

```python
first_p_tag = soup.find('p')
```

- **Finding All Tags:**

```python
all_p_tags = soup.find_all('p')
```

### **7. Accessing Tag Attributes:**

```python
link_tag = soup.find('a')
href_attribute = link_tag['href']
```

### **8. Extracting Text:**

```python
paragraph_text = first_p_tag.text
```

### **9. Filtering Content with CSS Selectors:**

```python
# Selecting by class
class_content = soup.select('.class-name')

# Selecting by ID
id_content = soup.select('#id-name')
```

### **10. Extracting Data from Tables:**

```python
table = soup.find('table')
rows = table.find_all('tr')

for row in rows:
    cells = row.find_all(['td', 'th'])
    for cell in cells:
        print(cell.text)
```

### **11. Handling Dynamic Content:**

- **Using Selenium for Dynamic Content:**

```python
from selenium import webdriver

driver = webdriver.Chrome()
driver.get(url)

# Get the page source after dynamic content is loaded
dynamic_content = driver.page_source

soup = BeautifulSoup(dynamic_content, 'html.parser')
```

### **12. Handling Pagination:**

```python
# Iterate through pages and scrape content
```

### **13. Saving Scraped Data:**

```python
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(soup.prettify())
```

### **14. Dealing with Errors:**

```python
try:
    # Web scraping code
except Exception as e:
    print(f"Error: {e}")
```

### **15. Advanced Features:**

- **Using Regular Expressions:**

```python
import re

pattern = re.compile(r'\d+')
matches = pattern.findall(text)
```

- **Handling Sessions:**

```python
import requests

with requests.Session() as session:
    response = session.get(url)
    # Continue with scraping...
```

This cheat sheet covers essential concepts for web scraping in Python using Beautiful Soup. Adjust the code snippets based on the specific structure of the websites you are scraping. Be aware of the terms of service of the website and ensure compliance with web scraping policies. Refer to the [official Beautiful Soup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) for more details and advanced features.

This Python cheat sheet for data engineers covers key libraries and frameworks for data processing, storage, ETL, serialization, visualization, and miscellaneous tasks. Adjust the code snippets based on your specific use cases and requirements.
