# Google Cloud Platform (GCP) Cheat Sheet for Data Engineers

## Google Cloud SDK Basics Cheat Sheet

### Installation and Configuration:

- **Install SDK:**
  ```bash
  gcloud components install
  ```

- **Update SDK:**
  ```bash
  gcloud components update
  ```

- **Authenticate User:**
  ```bash
  gcloud auth login
  ```

- **Set Project ID:**
  ```bash
  gcloud config set project [PROJECT-ID]
  ```

- **Display Configuration:**
  ```bash
  gcloud config list
  ```

### Resource Management:

- **List Projects:**
  ```bash
  gcloud projects list
  ```

- **List Compute Engine Instances:**
  ```bash
  gcloud compute instances list
  ```

- **List Cloud Storage Buckets:**
  ```bash
  gsutil ls
  ```

### Permissions and IAM:

- **List IAM Policies:**
  ```bash
  gcloud iam list-policies
  ```

- **Grant IAM Role to User:**
  ```bash
  gcloud projects add-iam-policy-binding [PROJECT-ID] --member="[MEMBER]" --role="[ROLE]"
  ```

### Networking:

- **List Firewall Rules:**
  ```bash
  gcloud compute firewall-rules list
  ```

- **List Forwarding Rules:**
  ```bash
  gcloud compute forwarding-rules list
  ```

### Command Line Output Format:

- **Set Default Output Format:**
  ```bash
  gcloud config set core/project [PROJECT-ID]
  ```

- **Set Default Output Format:**
  ```bash
  gcloud config set core/project [PROJECT-ID]
  ```

### SDK Version:

- **Display SDK Version:**
  ```bash
  gcloud version
  ```

### Additional Help:

- **Get Command-Specific Help:**
  ```bash
  gcloud [COMMAND] --help
  ```

- **Explore Available Commands:**
  ```bash
  gcloud
  ```

## Google Cloud Storage Cheat Sheet

### Bucket Operations:

- **Create Bucket:**
  ```bash
  gsutil mb gs://[BUCKET-NAME]
  ```

- **Remove Empty Bucket:**
  ```bash
  gsutil rb gs://[BUCKET-NAME]
  ```

- **List Buckets:**
  ```bash
  gsutil ls
  ```

### File Operations:

- **Copy Local File to Bucket:**
  ```bash
  gsutil cp [LOCAL-FILE] gs://[BUCKET-NAME]/
  ```

- **Copy Bucket Object to Local:**
  ```bash
  gsutil cp gs://[BUCKET-NAME]/[OBJECT-NAME] [LOCAL-DIRECTORY]
  ```

- **Move/Rename Object in Bucket:**
  ```bash
  gsutil mv gs://[BUCKET-NAME]/[OLD-OBJECT-NAME] gs://[BUCKET-NAME]/[NEW-OBJECT-NAME]
  ```

- **Remove Object from Bucket:**
  ```bash
  gsutil rm gs://[BUCKET-NAME]/[OBJECT-NAME]
  ```

### Access Control:

- **View Bucket ACL:**
  ```bash
  gsutil acl get gs://[BUCKET-NAME]
  ```

- **Set Bucket ACL:**
  ```bash
  gsutil acl ch -u [USER]:[PERMISSION] gs://[BUCKET-NAME]
  ```

- **View Object ACL:**
  ```bash
  gsutil acl get gs://[BUCKET-NAME]/[OBJECT-NAME]
  ```

- **Set Object ACL:**
  ```bash
  gsutil acl ch -u [USER]:[PERMISSION] gs://[BUCKET-NAME]/[OBJECT-NAME]
  ```

### Listing Contents:

- **List Contents of a Bucket:**
  ```bash
  gsutil ls gs://[BUCKET-NAME]
  ```

- **List Contents with Details:**
  ```bash
  gsutil ls -l gs://[BUCKET-NAME]
  ```

### Synchronization:

- **Sync Local Directory to Bucket:**
  ```bash
  gsutil rsync -r [LOCAL-DIRECTORY] gs://[BUCKET-NAME]
  ```

- **Sync Bucket to Local Directory:**
  ```bash
  gsutil rsync -r gs://[BUCKET-NAME] [LOCAL-DIRECTORY]
  ```

### Versioning:

- **Enable Versioning for a Bucket:**
  ```bash
  gsutil versioning set on gs://[BUCKET-NAME]
  ```

- **Suspend Versioning for a Bucket:**
  ```bash
  gsutil versioning set off gs://[BUCKET-NAME]
  ```

### Storage Class:

- **Set Storage Class for an Object:**
  ```bash
  gsutil storageclass set [STORAGE-CLASS] gs://[BUCKET-NAME]/[OBJECT-NAME]
  ```

- **Set Default Storage Class for a Bucket:**
  ```bash
  gsutil defstorageclass set [STORAGE-CLASS] gs://[BUCKET-NAME]
  ```

## Google BigQuery Cheat Sheet

### **Query Execution:**

- **Run SQL Query:**
  ```bash
  bq query --nouse_legacy_sql '[YOUR-SQL-QUERY]'
  ```

- **Save Query Results to a Table:**
  ```bash
  bq query --destination_table [PROJECT-ID]:[DATASET].[TABLE-NAME] --nouse_legacy_sql '[YOUR-SQL-QUERY]'
  ```

- **Export Query Results to CSV:**
  ```bash
  bq query --format=csv --nouse_legacy_sql '[YOUR-SQL-QUERY]' > [CSV-FILE]
  ```

### **Table Operations:**

- **List Tables in a Dataset:**
  ```bash
  bq ls [PROJECT-ID]:[DATASET]
  ```

- **Show Table Schema:**
  ```bash
  bq show --schema [PROJECT-ID]:[DATASET].[TABLE-NAME]
  ```

- **Load Data from CSV File:**
  ```bash
  bq load --source_format=CSV [PROJECT-ID]:[DATASET].[TABLE-NAME] [CSV-FILE-PATH] [SCHEMA]
  ```

- **Export Table to GCS:**
  ```bash
  bq extract [PROJECT-ID]:[DATASET].[TABLE-NAME] gs://[BUCKET]/[OBJECT]
  ```

### **Dataset Operations:**

- **List Datasets in a Project:**
  ```bash
  bq ls [PROJECT-ID]
  ```

- **Create Dataset:**
  ```bash
  bq mk [PROJECT-ID]:[DATASET]
  ```

- **Delete Dataset and its Contents:**
  ```bash
  bq rm -r [PROJECT-ID]:[DATASET]
  ```

### **Project Operations:**

- **Show Project Information:**
  ```bash
  bq show [PROJECT-ID]
  ```

- **Set Default Project:**
  ```bash
  bq mk --project_id [PROJECT-ID]
  ```

### **Job Operations:**

- **List Recent Jobs:**
  ```bash
  bq ls -j -a
  ```

- **Show Job Information:**
  ```bash
  bq show -j [JOB-ID]
  ```

### **Miscellaneous:**

- **Enable Standard SQL:**
  ```bash
  bq query --use_legacy_sql=false '[YOUR-SQL-QUERY]'
  ```

- **Enable Debug Mode:**
  ```bash
  bq query --nouse_legacy_sql --apilog=.[LOG-FILE] '[YOUR-SQL-QUERY]'
  ```

- **Display Help:**
  ```bash
  bq --help
  ```

## Google BigQuery SQL DDL Cheat Sheet

### **Creating Tables:**

- **Create a Table:**
  ```sql
  CREATE TABLE `project.dataset.new_table`
  (column1 datatype1, column2 datatype2, ...);
  ```

- **Create a Table with Options:**
  ```sql
  CREATE TABLE `project.dataset.new_table`
  (column1 datatype1, column2 datatype2, ...)
  OPTIONS(description="Table description", expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY));
  ```

### **Modifying Tables:**

- **Add a Column:**
  ```sql
  ALTER TABLE `project.dataset.table`
  ADD COLUMN new_column datatype;
  ```

- **Modify Column Data Type:**
  ```sql
  ALTER TABLE `project.dataset.table`
  ALTER COLUMN column1 SET DATA TYPE new_datatype;
  ```

- **Rename a Column:**
  ```sql
  ALTER TABLE `project.dataset.table`
  RENAME COLUMN old_column TO new_column;
  ```

- **Drop a Column:**
  ```sql
  ALTER TABLE `project.dataset.table`
  DROP COLUMN column_to_drop;
  ```

### **Partitioning and Clustering:**

- **Partition a Table:**
  ```sql
  CREATE TABLE `project.dataset.partitioned_table`
  PARTITION BY DATE(date_column)
  AS SELECT * FROM `project.dataset.table`;
  ```

- **Cluster a Table:**
  ```sql
  CREATE TABLE `project.dataset.clustered_table`
  CLUSTER BY column_to_cluster
  AS SELECT * FROM `project.dataset.table`;
  ```

### **Copying and Exporting Tables:**

- **Copy Table Within the Same Project:**
  ```sql
  CREATE TABLE `project.dataset.new_table`
  AS SELECT * FROM `project.dataset.source_table`;
  ```

- **Copy Table to Another Project:**
  ```sql
  CREATE TABLE `another_project.dataset.new_table`
  AS SELECT * FROM `project.dataset.source_table`;
  ```

- **Export Table to GCS:**
  ```sql
  EXPORT DATA OPTIONS(uri='gs://[BUCKET]/[OBJECT]', format='CSV')
  AS SELECT * FROM `project.dataset.table`;
  ```

### **Table Cloning:**

- **Clone Table Structure:**
  ```sql
  CREATE TABLE `project.dataset.new_table` AS
  SELECT * FROM `project.dataset.existing_table` WHERE FALSE;
  ```

### **Deleting Tables:**

- **Delete Table:**
  ```sql
  DROP TABLE `project.dataset.table_to_drop`;
  ```

- **Delete All Rows in a Table:**
  ```sql
  DELETE FROM `project.dataset.table`;
  ```

- **Delete Specific Rows Based on Condition:**
  ```sql
  DELETE FROM `project.dataset.table`
  WHERE condition;
  ```
## Google BigQuery SQL Cheat Sheet

### **Basic Queries:**

- **Select All Columns:**
  ```sql
  SELECT *
  FROM `project.dataset.table`
  ```

- **Select Specific Columns:**
  ```sql
  SELECT column1, column2
  FROM `project.dataset.table`
  ```

- **Filter Rows with WHERE Clause:**
  ```sql
  SELECT *
  FROM `project.dataset.table`
  WHERE condition
  ```

### **Aggregation Functions:**

- **Count Rows:**
  ```sql
  SELECT COUNT(*)
  FROM `project.dataset.table`
  ```

- **Sum of a Column:**
  ```sql
  SELECT SUM(column)
  FROM `project.dataset.table`
  ```

- **Average of a Column:**
  ```sql
  SELECT AVG(column)
  FROM `project.dataset.table`
  ```

### **Sorting and Limiting:**

- **Order by Column:**
  ```sql
  SELECT *
  FROM `project.dataset.table`
  ORDER BY column ASC/DESC
  ```

- **Limit Rows Returned:**
  ```sql
  SELECT *
  FROM `project.dataset.table`
  LIMIT n
  ```

### **Joins:**

- **Inner Join:**
  ```sql
  SELECT *
  FROM `project.dataset.table1` t1
  JOIN `project.dataset.table2` t2
  ON t1.column = t2.column
  ```

- **Left Join:**
  ```sql
  SELECT *
  FROM `project.dataset.table1` t1
  LEFT JOIN `project.dataset.table2` t2
  ON t1.column = t2.column
  ```

### **Grouping and Aggregations:**

- **Group by Column:**
  ```sql
  SELECT column, COUNT(*)
  FROM `project.dataset.table`
  GROUP BY column
  ```

- **Having Clause:**
  ```sql
  SELECT column, COUNT(*)
  FROM `project.dataset.table`
  GROUP BY column
  HAVING COUNT(*) > n
  ```

### **Subqueries:**

- **Simple Subquery:**
  ```sql
  SELECT *
  FROM `project.dataset.table`
  WHERE column IN (SELECT column FROM `project.dataset.table2`)
  ```

- **Correlated Subquery:**
  ```sql
  SELECT *
  FROM `project.dataset.table` t1
  WHERE column > (SELECT AVG(column) FROM `project.dataset.table` t2 WHERE t2.category = t1.category)
  ```
## Google BigQuery SQL Cheat Sheet: OVER, PARTITION BY, and WITH Clause

### **OVER Clause:**

- **Calculate Running Total:**
  ```sql
  SELECT column1, column2, SUM(column3) OVER (ORDER BY column1) AS running_total
  FROM `project.dataset.table`;
  ```

- **Calculate Moving Average:**
  ```sql
  SELECT column1, column2, AVG(column3) OVER (ORDER BY column1 ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
  FROM `project.dataset.table`;
  ```

- **Assign Row Numbers:**
  ```sql
  SELECT column1, column2, ROW_NUMBER() OVER (ORDER BY column1) AS row_number
  FROM `project.dataset.table`;
  ```

### **PARTITION BY Clause:**

- **Calculate Partitioned Aggregates:**
  ```sql
  SELECT column1, column2, SUM(column3) OVER (PARTITION BY column4 ORDER BY column1) AS partitioned_sum
  FROM `project.dataset.table`;
  ```

- **Assign Row Numbers Within Partitions:**
  ```sql
  SELECT column1, column2, ROW_NUMBER() OVER (PARTITION BY column3 ORDER BY column1) AS row_number_within_partition
  FROM `project.dataset.table`;
  ```

### **WITH Clause (Common Table Expressions - CTE):**

- **Create a Common Table Expression:**
  ```sql
  WITH cte_name AS (
    SELECT column1, column2
    FROM `project.dataset.table`
    WHERE condition
  )
  SELECT * FROM cte_name;
  ```

- **Use CTE in Multiple Queries:**
  ```sql
  WITH cte_name AS (
    SELECT column1, column2
    FROM `project.dataset.table`
    WHERE condition
  )
  SELECT * FROM cte_name WHERE column1 > 100;
  ```

- **Multiple CTEs in a Single Query:**
  ```sql
  WITH cte1 AS (SELECT ...),
       cte2 AS (SELECT ...)
  SELECT *
  FROM cte1
  JOIN cte2 ON cte1.column = cte2.column;
  ```

### **Combined Example:**

- **Calculate Average Sales per Region with CTE:**
  ```sql
  WITH region_sales AS (
    SELECT region, SUM(sales) AS total_sales
    FROM `project.dataset.sales_table`
    GROUP BY region
  )
  SELECT region, total_sales, AVG(total_sales) OVER () AS avg_sales_per_region
  FROM region_sales;
  ```
    
## Google BigQuery SQL DDL

### **Creating Tables:**

- **Create a Table:**
  ```sql
  CREATE TABLE `project.dataset.new_table`
  (column1 datatype1, column2 datatype2, ...);
  ```

- **Create a Table with Options:**
  ```sql
  CREATE TABLE `project.dataset.new_table`
  (column1 datatype1, column2 datatype2, ...)
  OPTIONS(description="Table description", expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY));
  ```

### **Modifying Tables:**

- **Add a Column:**
  ```sql
  ALTER TABLE `project.dataset.table`
  ADD COLUMN new_column datatype;
  ```

- **Modify Column Data Type:**
  ```sql
  ALTER TABLE `project.dataset.table`
  ALTER COLUMN column1 SET DATA TYPE new_datatype;
  ```

- **Rename a Column:**
  ```sql
  ALTER TABLE `project.dataset.table`
  RENAME COLUMN old_column TO new_column;
  ```

- **Drop a Column:**
  ```sql
  ALTER TABLE `project.dataset.table`
  DROP COLUMN column_to_drop;
  ```

### **Partitioning and Clustering:**

- **Partition a Table:**
  ```sql
  CREATE TABLE `project.dataset.partitioned_table`
  PARTITION BY DATE(date_column)
  AS SELECT * FROM `project.dataset.table`;
  ```

- **Cluster a Table:**
  ```sql
  CREATE TABLE `project.dataset.clustered_table`
  CLUSTER BY column_to_cluster
  AS SELECT * FROM `project.dataset.table`;
  ```

### **Copying and Exporting Tables:**

- **Copy Table Within the Same Project:**
  ```sql
  CREATE TABLE `project.dataset.new_table`
  AS SELECT * FROM `project.dataset.source_table`;
  ```

- **Copy Table to Another Project:**
  ```sql
  CREATE TABLE `another_project.dataset.new_table`
  AS SELECT * FROM `project.dataset.source_table`;
  ```

- **Export Table to GCS:**
  ```sql
  EXPORT DATA OPTIONS(uri='gs://[BUCKET]/[OBJECT]', format='CSV')
  AS SELECT * FROM `project.dataset.table`;
  ```

### **Table Cloning:**

- **Clone Table Structure:**
  ```sql
  CREATE TABLE `project.dataset.new_table` AS
  SELECT * FROM `project.dataset.existing_table` WHERE FALSE;
  ```

### **Deleting Tables:**

- **Delete Table:**
  ```sql
  DROP TABLE `project.dataset.table_to_drop`;
  ```

- **Delete All Rows in a Table:**
  ```sql
  DELETE FROM `project.dataset.table`;
  ```

- **Delete Specific Rows Based on Condition:**
  ```sql
  DELETE FROM `project.dataset.table`
  WHERE condition;
  ```
**Google BigQuery DML Cheat Sheet**

### **Inserting Data:**

- **Insert Rows Into a Table:**
  ```sql
  INSERT INTO `project.dataset.table` (column1, column2, ...)
  VALUES (value1, value2, ...), (value1, value2, ...), ...;
  ```

- **Insert Rows From Another Table:**
  ```sql
  INSERT INTO `project.dataset.table` (column1, column2, ...)
  SELECT column1, column2, ...
  FROM `project.dataset.source_table`
  WHERE condition;
  ```

### **Updating Data:**

- **Update Rows in a Table:**
  ```sql
  UPDATE `project.dataset.table`
  SET column1 = value1, column2 = value2, ...
  WHERE condition;
  ```

- **Update Using Subquery:**
  ```sql
  UPDATE `project.dataset.table`
  SET column1 = subquery.value1, column2 = subquery.value2, ...
  FROM (SELECT ... FROM `project.dataset.subquery_table`) subquery
  WHERE condition;
  ```

### **Deleting Data:**

- **Delete Rows From a Table:**
  ```sql
  DELETE FROM `project.dataset.table`
  WHERE condition;
  ```

- **Delete Using Subquery:**
  ```sql
  DELETE FROM `project.dataset.table`
  FROM (SELECT ... FROM `project.dataset.subquery_table`) subquery
  WHERE condition;
  ```

### **Merging Data (Upsert):**

- **Merging Rows Into a Table:**
  ```sql
  MERGE `project.dataset.target_table` T
  USING `project.dataset.source_table` S
  ON T.key_column = S.key_column
  WHEN MATCHED THEN
    UPDATE SET T.column1 = S.column1, T.column2 = S.column2, ...
  WHEN NOT MATCHED THEN
    INSERT (column1, column2, ...)
    VALUES (S.column1, S.column2, ...);
  ```

## Google Cloud Dataflow Cheat Sheet

### **Basic DataFlow Commands:**

- **Run a Dataflow Job:**
  ```bash
  gcloud dataflow jobs run [JOB-NAME] --gcs-location gs://dataflow-templates/latest/[TEMPLATE-NAME]
  ```

- **List Dataflow Jobs:**
  ```bash
  gcloud dataflow jobs list
  ```

### **Pipeline Building Blocks:**

- **Read Data From a Source:**
  ```python
  | 'ReadFromSource' >> ReadFromText('gs://your-bucket/your-input-file.txt')
  ```

- **Transform Data with ParDo:**
  ```python
  | 'TransformData' >> ParDo(YourCustomDoFn())
  ```

- **Write Data to a Sink:**
  ```python
  | 'WriteToSink' >> WriteToText('gs://your-bucket/your-output-file.txt')
  ```

### **Windowing and Time Operations:**

- **Fixed Windows:**
  ```python
  | 'FixedWindows' >> WindowInto(FixedWindows(window_size))
  ```

- **Sliding Windows:**
  ```python
  | 'SlidingWindows' >> WindowInto(SlidingWindows(window_size, offset))
  ```

- **Windowed ParDo:**
  ```python
  | 'WindowedTransform' >> ParDo(YourWindowedDoFn())
  ```

### **Joins and GroupBy:**

- **CoGroupByKey for Join:**
  ```python
  | 'JoinCollections' >> CoGroupByKey()
  ```

- **GroupByKey for Aggregation:**
  ```python
  | 'GroupBy' >> GroupByKey()
  ```

### **Side Inputs and Outputs:**

- **Define Side Input:**
  ```python
  side_input = p | 'ReadSideInput' >> ReadFromText('gs://your-bucket/side-input.txt')
  ```

- **Use Side Input in DoFn:**
  ```python
  class YourDoFn(beam.DoFn):
      def process(self, element, side_input):
          # Your logic here
  ```

### **Windowing and Triggers:**

- **Set Trigger on Windowed Data:**
  ```python
  | 'TriggerWindow' >> Repeatedly(AfterCount(trigger_count)).\
                       UntilCount(trigger_count)
  ```

### **Custom Transforms and DoFns:**

- **Create a Custom Transform:**
  ```python
  class YourTransform(beam.PTransform):
      def expand(self, pcoll):
          # Your logic here
  ```

- **Create a Custom DoFn:**
  ```python
  class YourDoFn(beam.DoFn):
      def process(self, element):
          # Your logic here
  ```

### **Running Locally for Development:**

- **Run Dataflow Locally:**
  ```bash
  python your_pipeline.py --runner=DirectRunner
  ```

- **Use Local Files as Input/Output:**
  ```python
  | 'ReadLocalFile' >> ReadFromText('/path/to/local/file.txt')
  ```

### Pub/Sub:

**Google Cloud Pub/Sub Cheat Sheet**

### **Basic Pub/Sub Commands:**

- **Create a Topic:**
  ```bash
  gcloud pubsub topics create [TOPIC-NAME]
  ```

- **List Topics:**
  ```bash
  gcloud pubsub topics list
  ```

- **Delete a Topic:**
  ```bash
  gcloud pubsub topics delete [TOPIC-NAME]
  ```

- **Create a Subscription:**
  ```bash
  gcloud pubsub subscriptions create [SUBSCRIPTION-NAME] --topic [TOPIC-NAME]
  ```

- **List Subscriptions:**
  ```bash
  gcloud pubsub subscriptions list
  ```

- **Delete a Subscription:**
  ```bash
  gcloud pubsub subscriptions delete [SUBSCRIPTION-NAME]
  ```

### **Publish and Consume Messages:**

- **Publish a Message:**
  ```bash
  gcloud pubsub topics publish [TOPIC-NAME] --message "[YOUR-MESSAGE]"
  ```

- **Pull Messages from a Subscription:**
  ```bash
  gcloud pubsub subscriptions pull [SUBSCRIPTION-NAME] --auto-ack
  ```

- **Acknowledge a Message:**
  ```bash
  gcloud pubsub subscriptions acknowledge [SUBSCRIPTION-NAME] "[ACK-ID]"
  ```

### **Batch Operations:**

- **Publish Batch of Messages:**
  ```bash
  gcloud pubsub topics publish [TOPIC-NAME] --message-file messages.txt
  ```

- **Pull Batch of Messages:**
  ```bash
  gcloud pubsub subscriptions pull [SUBSCRIPTION-NAME] --max-messages=10 --auto-ack
  ```

### **Flow Control:**

- **Set Acknowledge Deadline:**
  ```bash
  gcloud pubsub subscriptions modify [SUBSCRIPTION-NAME] --ack-deadline=60
  ```

### **Snapshot Operations:**

- **Create a Snapshot:**
  ```bash
  gcloud pubsub snapshots create [SNAPSHOT-NAME] --subscription [SUBSCRIPTION-NAME]
  ```

- **List Snapshots:**
  ```bash
  gcloud pubsub snapshots list
  ```

- **Delete a Snapshot:**
  ```bash
  gcloud pubsub snapshots delete [SNAPSHOT-NAME]
  ```

### **IAM Permissions:**

- **Grant Pub/Sub Permissions:**
  ```bash
  gcloud pubsub topics add-iam-policy-binding [TOPIC-NAME] --member="[MEMBER]" --role="[ROLE]"
  ```

- **Revoke Pub/Sub Permissions:**
  ```bash
  gcloud pubsub topics remove-iam-policy-binding [TOPIC-NAME] --member="[MEMBER]" --role="[ROLE]"
  ```

### **Push Subscriptions:**

- **Create a Push Subscription:**
  ```bash
  gcloud pubsub subscriptions create [SUBSCRIPTION-NAME] --topic [TOPIC-NAME] --push-endpoint=[PUSH-ENDPOINT]
  ```

- **Modify Push Endpoint:**
  ```bash
  gcloud pubsub subscriptions modify [SUBSCRIPTION-NAME] --push-endpoint=[NEW-PUSH-ENDPOINT]
  ```

## Google Cloud Composer (Airflow)

### **Environment Management:**

- **Create a Composer Environment:**
  ```bash
  gcloud composer environments create [ENVIRONMENT-NAME] --location [REGION] --project [PROJECT-ID]
  ```

- **List Composer Environments:**
  ```bash
  gcloud composer environments list --locations [REGION] --project [PROJECT-ID]
  ```

- **Delete a Composer Environment:**
  ```bash
  gcloud composer environments delete [ENVIRONMENT-NAME] --location [REGION] --project [PROJECT-ID]
  ```

### **DAG (Directed Acyclic Graph) Management:**

- **Upload a DAG File:**
  ```bash
  gcloud composer environments storage dags import --environment [ENVIRONMENT-NAME] --location [REGION] --source [DAG-FILE]
  ```

- **List DAGs:**
  ```bash
  gcloud composer environments storage dags list --environment [ENVIRONMENT-NAME] --location [REGION]
  ```

- **Delete a DAG:**
  ```bash
  gcloud composer environments storage dags delete [DAG-NAME] --environment [ENVIRONMENT-NAME] --location [REGION]
  ```

### **Triggering and Managing DAGs:**

- **Trigger a DAG Run:**
  ```bash
  gcloud composer environments run [ENVIRONMENT-NAME] --location [REGION] trigger_dag -- [DAG-NAME]
  ```

- **List DAG Runs:**
  ```bash
  gcloud composer environments runs list --environment [ENVIRONMENT-NAME] --location [REGION] -- [DAG-NAME]
  ```

- **View DAG Run Details:**
  ```bash
  gcloud composer environments runs describe [RUN-ID] --environment [ENVIRONMENT-NAME] --location [REGION] -- [DAG-NAME]
  ```

### **IAM and Permissions:**

- **Grant Permissions to a User:**
  ```bash
  gcloud composer environments add-permissions [ENVIRONMENT-NAME] --location [REGION] --member [MEMBER] --role [ROLE]
  ```

- **Revoke Permissions from a User:**
  ```bash
  gcloud composer environments remove-permissions [ENVIRONMENT-NAME] --location [REGION] --member [MEMBER] --role [ROLE]
  ```

### **Scaling and Updating Environment Configuration:**

- **Update Environment Configuration:**
  ```bash
  gcloud composer environments update [ENVIRONMENT-NAME] --location [REGION] --update-env-variables [KEY=VALUE,...]
  ```

- **Set Autoscaling Parameters:**
  ```bash
  gcloud composer environments update [ENVIRONMENT-NAME] --location [REGION] --update-env-variables [AUTOSCALING-CONFIG]
  ```

### **Accessing Airflow Web Interface:**

- **Open Airflow Web Interface:**
  ```bash
  gcloud composer environments run [ENVIRONMENT-NAME] --location [REGION] webserver -- [DAG-NAME]
  ```

### **Additional Operations:**

- **List Available Image Versions:**
  ```bash
  gcloud composer versions list
  ```

- **Upgrade Composer Environment Image Version:**
  ```bash
  gcloud composer environments update [ENVIRONMENT-NAME] --location [REGION] --image-version [IMAGE-VERSION]
  ```
It seems there might be a confusion in your request. If you meant Google Cloud Composer, it is a fully managed workflow orchestration service built on Apache Airflow. If you were referring to another service or tool, please provide clarification.

Assuming you meant Google Cloud Composer, below is a Python cheat sheet for Google Cloud Composer:

**Google Cloud Composer Python Cheat Sheet**

### **Environment and DAG Management:**

- **Create a Composer Environment:**
  ```python
  from google.cloud import composer_v1
  from google.cloud.composer_v1 import types

  client = composer_v1.EnvironmentsClient()
  location = "your-region"
  project_id = "your-project-id"
  environment = types.Environment(
      name="projects/{}/locations/{}/environments/{}".format(
          project_id, location, "your-environment-name"
      ),
      config={
          "node_count": 3,
          "software_config": {"image_version": "composer-2.0.0-preview.6"},
      },
  )
  operation = client.create_environment(parent="projects/{}/locations/{}".format(project_id, location), environment=environment)
  result = operation.result()
  print(f"Environment created successfully: {result.name}")
  ```

- **List Composer Environments:**
  ```python
  environments = client.list_environments(parent="projects/{}/locations/{}".format(project_id, location))
  for environment in environments:
      print(f"Environment Name: {environment.name}")
  ```

- **Create a DAG File:**
  ```python
  # Upload your DAG file to a GCS bucket
  dag_file_path = "gs://your-bucket/your-dag.py"

  # Set the DAG properties
  dag = {
      "name": "your-dag-name",
      "source_code_uri": dag_file_path,
      "dag_airflow_config": {"example_key": "example_value"},
  }

  # Create the DAG
  client = composer_v1.ImageVersionsClient()
  client.create_dag(parent="projects/{}/locations/{}/environments/{}".format(project_id, location, "your-environment-name"), dag=dag)
  ```

### **Triggering and Managing DAGs:**

- **Trigger a DAG Run:**
  ```python
  client = composer_v1.EnvironmentsClient()
  client.run_dag(name="projects/{}/locations/{}/environments/{}/dags/{}".format(project_id, location, "your-environment-name", "your-dag-name"))
  ```

- **List DAG Runs:**
  ```python
  client = composer_v1.EnvironmentsClient()
  dag_runs = client.list_dag_runs(parent="projects/{}/locations/{}/environments/{}/dags/{}".format(project_id, location, "your-environment-name", "your-dag-name"))
  for dag_run in dag_runs:
      print(f"DAG Run ID: {dag_run.dag_run_id}")
  ```

- **View DAG Run Details:**
  ```python
  client = composer_v1.EnvironmentsClient()
  dag_run = client.get_dag_run(name="projects/{}/locations/{}/environments/{}/dags/{}/dagRuns/{}".format(project_id, location, "your-environment-name", "your-dag-name", "your-dag-run-id"))
  print(f"DAG Run Details: {dag_run}")
  ```

### **IAM and Permissions:**

- **Grant Permissions to a User:**
  ```python
  client = composer_v1.EnvironmentsClient()
  client.add_environment_variable(
      name="projects/{}/locations/{}/environments/{}/variables/{}".format(project_id, location, "your-environment-name", "your-variable-name"),
      environment_variable=types.EnvironmentVariable(
          name="projects/{}/locations/{}/environments/{}/variables/{}".format(project_id, location, "your-environment-name", "your-variable-name"),
          value="your-variable-value",
      ),
  )
  ```

- **Revoke Permissions from a User:**
  ```python
  client = composer_v1.EnvironmentsClient()
  client.delete_environment_variable(name="projects/{}/locations/{}/environments/{}/variables/{}".format(project_id, location, "your-environment-name", "your-variable-name"))
  ```

### **Additional Operations:**

- **List Available Composer Image Versions:**
  ```python
  client = composer_v1.ImageVersionsClient()
  image_versions = client.list_image_versions(parent="projects/{}/locations/{}".format(project_id, location))
  for image_version in image_versions:
      print(f"Image Version: {image_version.image_version}")
  ```

- **Upgrade Composer Environment Image Version:**
  ```python
  client = composer_v1.EnvironmentsClient()
  client.update_environment(name="projects/{}/locations/{}/environments/{}/imageVersions/{}".format(project_id, location, "your-environment-name", "your-image-version"))
  ```

## Google Cloud Dataproc

### **Cluster Management:**

- **Create a Dataproc Cluster:**
  ```bash
  gcloud dataproc clusters create [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID] --num-workers [NUM-WORKERS]
  ```

- **List Dataproc Clusters:**
  ```bash
  gcloud dataproc clusters list --region [REGION] --project [PROJECT-ID]
  ```

- **View Cluster Details:**
  ```bash
  gcloud dataproc clusters describe [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID]
  ```

- **Update Cluster Configuration:**
  ```bash
  gcloud dataproc clusters update [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID] --num-workers [NEW-NUM-WORKERS]
  ```

- **Delete a Dataproc Cluster:**
  ```bash
  gcloud dataproc clusters delete [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID]
  ```

### **Jobs and Workflows:**

- **Submit a PySpark Job:**
  ```bash
  gcloud dataproc jobs submit pyspark [JOB-FILE] --cluster [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID]
  ```

- **Submit a SparkSQL Job:**
  ```bash
  gcloud dataproc jobs submit sparksql --cluster [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID] --query-file [QUERY-FILE]
  ```

- **Submit a Pig Job:**
  ```bash
  gcloud dataproc jobs submit pig [JOB-FILE] --cluster [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID]
  ```

- **Submit a Hive Job:**
  ```bash
  gcloud dataproc jobs submit hive [JOB-FILE] --cluster [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID]
  ```

- **Submit a Spark Job with JAR:**
  ```bash
  gcloud dataproc jobs submit spark --cluster [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID] --class [MAIN-CLASS] --jars [JAR-FILE]
  ```

### **Job Management:**

- **List Dataproc Jobs:**
  ```bash
  gcloud dataproc jobs list --region [REGION] --project [PROJECT-ID]
  ```

- **View Job Details:**
  ```bash
  gcloud dataproc jobs describe [JOB-ID] --region [REGION] --project [PROJECT-ID]
  ```

- **Cancel a Dataproc Job:**
  ```bash
  gcloud dataproc jobs cancel [JOB-ID] --region [REGION] --project [PROJECT-ID]
  ```

### **Initialization Actions:**

- **Add Initialization Action to Cluster:**
  ```bash
  gcloud dataproc clusters update [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID] --update=[INIT-ACTION-SCRIPT]
  ```

### **IAM and Permissions:**

- **Grant Permissions to a User:**
  ```bash
  gcloud dataproc clusters add-iam-policy-binding [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID] --member [MEMBER] --role [ROLE]
  ```

- **Revoke Permissions from a User:**
  ```bash
  gcloud dataproc clusters remove-iam-policy-binding [CLUSTER-NAME] --region [REGION] --project [PROJECT-ID] --member [MEMBER] --role [ROLE]
  ```

### Cloud Storage Transfer Service:

## Google Cloud Storage Transfer Service

- **Create a Transfer Operation:**
  ```bash
  gcloud beta transfer operations create --config [TRANSFER-JOB-CONFIG-FILE] --project [PROJECT-ID]
  ```

- **List Transfer Operations:**
  ```bash
  gcloud beta transfer operations list --project [PROJECT-ID]
  ```

- **Pause a Transfer Operation:**
  ```bash
  gcloud beta transfer operations pause [OPERATION-NAME] --project [PROJECT-ID]
  ```

- **Resume a Transfer Operation:**
  ```bash
  gcloud beta transfer operations resume [OPERATION-NAME] --project [PROJECT-ID]
  ```

- **Cancel a Transfer Operation:**
  ```bash
  gcloud beta transfer operations cancel [OPERATION-NAME] --project [PROJECT-ID]
  ```
**Google Cloud Dataproc Python Cheat Sheet**

### **Google Cloud SDK Installation:**

- **Install Google Cloud SDK:**
  Follow the instructions in the [official documentation](https://cloud.google.com/sdk/docs/install) to install the Google Cloud SDK.

- **Authenticate with Google Cloud:**
  ```bash
  gcloud auth login
  ```

### **Cluster Management with Python:**

- **Create a Dataproc Cluster:**
  ```python
  from google.cloud import dataproc_v1
  from google.cloud.dataproc_v1 import enums

  client = dataproc_v1.ClusterControllerClient()
  project_id = "your-project-id"
  region = "your-region"

  cluster = {
      "project_id": project_id,
      "cluster_name": "your-cluster-name",
      "config": {
          "master_config": {"num_instances": 1, "machine_type_uri": "n1-standard-4"},
          "worker_config": {"num_instances": 2, "machine_type_uri": "n1-standard-4"},
      },
  }

  operation = client.create_cluster(
      project_id=project_id, region=region, cluster=cluster
  )
  result = operation.result()
  print(f"Cluster created successfully: {result.cluster_name}")
  ```

- **List Dataproc Clusters:**
  ```python
  clusters = client.list_clusters(project_id=project_id, region=region)
  for cluster in clusters:
      print(f"Cluster Name: {cluster.cluster_name}")
  ```

- **Delete a Dataproc Cluster:**
  ```python
  client.delete_cluster(project_id=project_id, region=region, cluster_name="your-cluster-name")
  ```

### **Job Submission with Python:**

- **Submit a PySpark Job:**
  ```python
  job = {
      "reference": {"project_id": project_id, "job_id": "your-job-id"},
      "placement": {"cluster_name": "your-cluster-name"},
      "pyspark_job": {"main_python_file_uri": "gs://your-bucket/your-script.py"},
  }

  operation = client.submit_job(project_id=project_id, region=region, job=job)
  result = operation.result()
  print(f"Job submitted successfully: {result.reference.job_id}")
  ```

- **List Dataproc Jobs:**
  ```python
  jobs = client.list_jobs(project_id=project_id, region=region)
  for job in jobs:
      print(f"Job ID: {job.reference.job_id}")
  ```

- **Cancel a Dataproc Job:**
  ```python
  client.cancel_job(
      project_id=project_id, region=region, job_id="your-job-id"
  )
  ```

### **Cluster and Job Configuration Files:**

- **Example Cluster Configuration JSON:**
  ```json
  {
      "project_id": "your-project-id",
      "cluster_name": "your-cluster-name",
      "config": {
          "master_config": {"num_instances": 1, "machine_type_uri": "n1-standard-4"},
          "worker_config": {"num_instances": 2, "machine_type_uri": "n1-standard-4"},
      },
  }
  ```

- **Example PySpark Job Configuration JSON:**
  ```json
  {
      "reference": {"project_id": "your-project-id", "job_id": "your-job-id"},
      "placement": {"cluster_name": "your-cluster-name"},
      "pyspark_job": {"main_python_file_uri": "gs://your-bucket/your-script.py"},
  }
  ```

### **Additional Python Libraries:**

- **Install Required Libraries:**
  ```bash
  pip install google-cloud-dataproc google-cloud-storage
  ```

### **Transfer Jobs:**

- **Create a Transfer Job:**
  ```bash
  gcloud beta transfer jobs create [JOB-NAME] --description "[DESCRIPTION]" --project [PROJECT-ID] --schedule [SCHEDULE] --transfer-spec [TRANSFER-SPEC-FILE]
  ```

- **List Transfer Jobs:**
  ```bash
  gcloud beta transfer jobs list --project [PROJECT-ID]
  ```

- **Get Transfer Job Details:**
  ```bash
  gcloud beta transfer jobs describe [JOB-NAME] --project [PROJECT-ID]
  ```

- **Update Transfer Job:**
  ```bash
  gcloud beta transfer jobs update [JOB-NAME] --project [PROJECT-ID] --description "[NEW-DESCRIPTION]" --transfer-spec [NEW-TRANSFER-SPEC-FILE]
  ```

- **Delete Transfer Job:**
  ```bash
  gcloud beta transfer jobs delete [JOB-NAME] --project [PROJECT-ID]
  ```

### **Transfer Spec Configuration:**

- **Example Transfer Spec JSON:**
  ```json
  {
    "gcsDataSource": {
      "bucketName": "[SOURCE-BUCKET]",
      "path": "[SOURCE-PATH]"
    },
    "gcsDataSink": {
      "bucketName": "[DESTINATION-BUCKET]",
      "path": "[DESTINATION-PATH]"
    }
  }
  ```

### **IAM and Permissions:**

- **Grant Permissions to a User:**
  ```bash
  gcloud beta transfer jobs add-iam-policy-binding [JOB-NAME] --project [PROJECT-ID] --member [MEMBER] --role [ROLE]
  ```

- **Revoke Permissions from a User:**
  ```bash
  gcloud beta transfer jobs remove-iam-policy-binding [JOB-NAME] --project [PROJECT-ID] --member [MEMBER] --role [ROLE]
  ```

### **Additional Operations:**

- **Check Operation Status:**
  ```bash
  gcloud beta transfer operations check --project [PROJECT-ID] [OPERATION-NAME]
  ```

- **List Transfer Operation Counters:**
  ```bash
  gcloud beta transfer operations get-counters --project [PROJECT-ID] [OPERATION-NAME]
  ```
## Google Cloud AI Platform (formerly ML Engine)

### **1. Training a Model:**

- **Using gcloud CLI:**
  ```bash
  gcloud ai-platform jobs submit training JOB_NAME \
    --module-name=your_module \
    --package-path=your_package \
    --staging-bucket=gs://your_bucket \
    --region=your_region \
    --runtime-version=your_version \
    --python-version=your_python_version \
    -- \
    --arg1=value1 \
    --arg2=value2
  ```

### **2. Deploying a Model:**

- **Using gcloud CLI:**
  ```bash
  gcloud ai-platform models create your_model \
    --regions=your_region
  ```

- **Deploying a Version:**
  ```bash
  gcloud ai-platform versions create your_version \
    --model=your_model \
    --origin=gs://your_model_directory \
    --runtime-version=your_version \
    --python-version=your_python_version
  ```

### **3. Online Prediction:**

- **Using gcloud CLI:**
  ```bash
  gcloud ai-platform predict \
    --model=your_model \
    --version=your_version \
    --json-instances=your_input.json
  ```

### **4. Batch Prediction:**

- **Using gcloud CLI:**
  ```bash
  gcloud ai-platform jobs submit prediction JOB_NAME \
    --model=your_model \
    --version=your_version \
    --data-format=TEXT \
    --region=your_region \
    --input-paths=gs://your_input_directory/*.json \
    --output-path=gs://your_output_directory
  ```

### **5. Hyperparameter Tuning:**

- **Using gcloud CLI:**
  ```bash
  gcloud ai-platform jobs submit training JOB_NAME \
    --module-name=your_module \
    --package-path=your_package \
    --staging-bucket=gs://your_bucket \
    --region=your_region \
    --runtime-version=your_version \
    --python-version=your_python_version \
    --config=your_hyperparameter_config.yaml
  ```

### **6. Model Versioning:**

- **Creating a Model Version:**
  ```bash
  gcloud ai-platform versions create your_version \
    --model=your_model \
    --origin=gs://your_model_directory \
    --runtime-version=your_version \
    --python-version=your_python_version
  ```

### **7. Model Monitoring:**

- **Configuring Model Monitoring:**
  Update the deployed model version with monitoring options.

### **8. Access Control:**

- **Setting Permissions:**
  Use Google Cloud Identity and Access Management (IAM) to manage permissions.

### **9. Cleaning Up:**

- **Deleting a Model:**
  ```bash
  gcloud ai-platform models delete your_model
  ```

- **Deleting a Version:**
  ```bash
  gcloud ai-platform versions delete your_version --model=your_model
  ```

### **10. Monitoring and Logging:**

- **Viewing Logs:**
  Access logs through Google Cloud Console or use Stackdriver Logging.

### **11. Best Practices:**

- **Use TensorBoard for Monitoring:**
  Visualize TensorFlow training runs using TensorBoard.

- **Optimize Input Data:**
  Preprocess and optimize input data for better model performance.

- **Regularly Monitor and Retrain:**
  Monitor model performance and retrain as needed for evolving data.

### **12. TensorFlow Extended (TFX):**

- **Building TFX Pipelines:**
  Leverage TensorFlow Extended for end-to-end ML pipelines.

**Note:** Ensure that you replace placeholders such as `[...]` with your specific values when using these commands. This cheat sheet provides a quick reference for common GCP commands used by data engineers. Always refer to the official GCP documentation for the most up-to-date information and additional details: [Google Cloud SDK Documentation](https://cloud.google.com/sdk/docs) and [Google Cloud Command-Line Tool Documentation](https://cloud.google.com/sdk/gcloud).
