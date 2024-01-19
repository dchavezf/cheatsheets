**Google Cloud Platform (GCP) Cheat Sheet for Data Engineers**

### Google Cloud SDK Basics:

- **Install SDK:**
  ```
  gcloud components install
  ```

- **Configure Credentials:**
  ```
  gcloud auth login
  ```

### Google Cloud Storage:

- **Copy Local File to Cloud Storage:**
  ```
  gsutil cp [LOCAL-FILE] gs://[DESTINATION-BUCKET]/
  ```

- **Copy Cloud Storage File to Local:**
  ```
  gsutil cp gs://[SOURCE-BUCKET]/[REMOTE-FILE] [LOCAL-PATH]
  ```

- **List Contents of a Bucket:**
  ```
  gsutil ls gs://[BUCKET-NAME]
  ```

### BigQuery:

- **Run SQL Query:**
  ```
  bq query --nouse_legacy_sql '[YOUR-SQL-QUERY]'
  ```

- **Export Query Result to CSV:**
  ```
  bq query --format=csv --nouse_legacy_sql '[YOUR-SQL-QUERY]' > [CSV-FILE]
  ```

- **Load Data from CSV File:**
  ```
  bq load --source_format=CSV [PROJECT-ID]:[DATASET].[TABLE-NAME] [CSV-FILE-PATH] [SCHEMA]
  ```

### Dataflow:

- **Run Dataflow Job:**
  ```
  gcloud dataflow jobs run [JOB-NAME] --gcs-location gs://dataflow-templates/latest/[TEMPLATE-NAME]
  ```

- **List Dataflow Jobs:**
  ```
  gcloud dataflow jobs list
  ```

### Dataprep:

- **Create a Dataprep Flow:**
  ```
  gcloud dataprep flows create --name="[FLOW-NAME]" --project="[PROJECT-ID]"
  ```

- **List Dataprep Flows:**
  ```
  gcloud dataprep flows list --project="[PROJECT-ID]"
  ```

### Pub/Sub:

- **Publish a Message:**
  ```
  gcloud pubsub topics publish [TOPIC-NAME] --message "[YOUR-MESSAGE]"
  ```

- **Subscribe to a Topic:**
  ```
  gcloud pubsub subscriptions create [SUBSCRIPTION-NAME] --topic [TOPIC-NAME]
  ```

### Cloud Composer (Airflow):

- **Trigger a DAG (Directed Acyclic Graph) Run:**
  ```
  gcloud composer environments run [ENVIRONMENT-NAME] --location [LOCATION] trigger_dag -- [DAG-NAME]
  ```

- **List DAGs:**
  ```
  gcloud composer environments run [ENVIRONMENT-NAME] --location [LOCATION] list_dags
  ```

### Cloud Storage Transfer Service:

- **Create Transfer Job:**
  ```
  gcloud beta transfer jobs create [JOB-NAME] --project=[PROJECT-ID] --description="[JOB-DESCRIPTION]" --schedule="[SCHEDULE]" --transfer-spec=gs://[SOURCE-BUCKET]/[PREFIX] gs://[DESTINATION-BUCKET]/[PREFIX]
  ```

- **List Transfer Jobs:**
  ```
  gcloud beta transfer jobs list --project=[PROJECT-ID]
  ```

**Note:** Ensure that you replace placeholders such as `[...]` with your specific values when using these commands. This cheat sheet provides a quick reference for common GCP commands used by data engineers. Always refer to the official GCP documentation for the most up-to-date information and additional details: [Google Cloud SDK Documentation](https://cloud.google.com/sdk/docs) and [Google Cloud Command-Line Tool Documentation](https://cloud.google.com/sdk/gcloud).
