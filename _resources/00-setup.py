# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./00-global-setup-bundle $reset_all_data=$reset_all_data $db_prefix=telco $min_dbr_version=11

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from mlflow import MlflowClient
import requests
from io import StringIO

# COMMAND ----------

#Dataset under apache license: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE
cust_csv = requests.get("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv").text
cust_df = pd.read_csv(StringIO(cust_csv), sep=",")

def cleanup_column(pdf):
  # Clean up column names
  pdf.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower().replace("__", "_") for name in pdf.columns]
  pdf.columns = [re.sub(r'[\(\)]', '', name).lower() for name in pdf.columns]
  pdf.columns = [re.sub(r'[ -]', '_', name).lower() for name in pdf.columns]
  return pdf.rename(columns = {'streaming_t_v': 'streaming_tv', 'customer_i_d': 'customer_id'})
  
cust_df = cleanup_column(cust_df)

mobile_csv = requests.get("https://raw.githubusercontent.com/nuwan-db/Graph_Analytics_Telco_Churn_Prediction/dev/_resources/data/telco_customer_mobile.csv").text
mobile_df = pd.read_csv(StringIO(mobile_csv), sep=",")

df = cust_df.merge(mobile_df, on='customer_id', how='left')
df = df[['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
       'tenure', 'phone_service', 'multiple_lines', 'internet_service',
       'online_security', 'online_backup', 'device_protection', 'tech_support',
       'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
       'payment_method', 'monthly_charges', 'total_charges', 'mobile_number', 'churn']]
spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("telco_churn_customers_bronze")

# COMMAND ----------

call_csv = requests.get("https://raw.githubusercontent.com/nuwan-db/Graph_Analytics_Telco_Churn_Prediction/dev/_resources/data/telco_call_log.csv").text
call_df = pd.read_csv(StringIO(call_csv), sep=",")
call_df = call_df[['datatime', 'caller_mobile_number', 'callee_mobile_number', 'duration']]
spark.createDataFrame(call_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("telco_call_log_bronze")

# COMMAND ----------

def display_automl_churn_link(table_name, force_refresh = False): 
  if force_refresh:
    reset_automl_run("churn_auto_ml")
  display_automl_link("churn_auto_ml", "demos_customer_churn", spark.table(table_name), "churn", 5)

def get_automl_churn_run(table_name = "churn_features", force_refresh = False): 
  if force_refresh:
    reset_automl_run("churn_auto_ml")
  from_cache, r = get_automl_run_or_start("churn_auto_ml", "demos_customer_churn", spark.table(table_name), "churn", 5)
  return r

# COMMAND ----------

# MAGIC %run ./API_Helpers

# COMMAND ----------

 
