# Databricks notebook source
# MAGIC %run ./_resources/00-setup $reset_all_data=false

# COMMAND ----------

from graphframes import *
from math import comb
import pyspark.sql.functions as F
from warnings import filterwarnings
filterwarnings('ignore', 'DataFrame.sql_ctx is an internal property')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engineering graph features
# MAGIC 
# MAGIC Graph features are generated using Spark GraphFrames with vertex and edge dataframes created in the [EDA](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/3631133442664989/command/3631133442977060) step.

# COMMAND ----------

# DBTITLE 1,Read vertex_df
vertex_df = spark.table("telco_vertex_df")
display(vertex_df)

# COMMAND ----------

# DBTITLE 1,Read edge_df
edge_df = spark.table("telco_edge_df")
display(edge_df)

# COMMAND ----------

# DBTITLE 1,Creating a graph using GraphFrames
g = GraphFrame(vertex_df, edge_df)

# COMMAND ----------

# DBTITLE 1,Degree
# Calculating the number of edges that are connected to each vertex.

degree_df = g.degrees
graph_features_df = vertex_df.alias('customer').join(degree_df, degree_df.id == vertex_df.id, 'left')\
                             .select('customer.id', 'degree')\
                             .withColumnRenamed('id','customer_id')\
                             .fillna(0, "degree")
          
display(graph_features_df.orderBy(col("degree").desc()))

# COMMAND ----------

# DBTITLE 1,In-degree
# Calculating the number of edges that are directed towards each vertex.

indegree_df = g.inDegrees
graph_features_df = graph_features_df.alias('features').join(indegree_df, indegree_df.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'inDegree')\
                 .fillna(0, "inDegree")\
                 .withColumnRenamed("inDegree","in_degree")
display(graph_features_df.orderBy(col("inDegree").desc()))

# COMMAND ----------

# DBTITLE 1,Out-degree
# Calculating the number of edges that are originated from each vertex.

outdegree_df = g.outDegrees
graph_features_df = graph_features_df.alias('features').join(outdegree_df, outdegree_df.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'outDegree')\
                 .fillna(0, "outDegree")\
                 .withColumnRenamed("outDegree","out_degree")
display(graph_features_df.orderBy(col("outDegree").desc()))

# COMMAND ----------

# DBTITLE 1,Degree ratio
# Calculating degree ratio

def degreeRatio(x, d):
  if d==0:
    return 0.0
  else:
    return x/d
  
degreeRatioUDF = udf(degreeRatio, FloatType())   

graph_features_df = graph_features_df.withColumn("in_degree_ratio", degreeRatioUDF(col("in_degree"), col("degree")))
graph_features_df = graph_features_df.withColumn("out_degree_ratio", degreeRatioUDF(col("out_degree"), col("degree")))
display(graph_features_df)

# COMMAND ----------

# DBTITLE 1,PageRank
# Calculating pagerank

pr_df = g.pageRank(resetProbability=0.15, tol=0.01).vertices.select('id','pagerank')
graph_features_df = graph_features_df.alias('features').join(pr_df, pr_df.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'pagerank')

display(graph_features_df)

# COMMAND ----------

# DBTITLE 1,Triangle Count
# Calculating triangle count

trian_count = g.triangleCount()

graph_features_df = graph_features_df.alias('features').join(trian_count.select('id','count'), trian_count.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'count')\
                 .withColumnRenamed("count","trian_count")

display(graph_features_df.orderBy(col("trian_count").desc()))

# COMMAND ----------

# DBTITLE 1,Clustering coefficient 
# Calculating clustering coefficient 

def clusterCoefficient(t, e):
  if e==0 or t==0:
    return 0.0
  else:
    return t/comb(e, 2)
  
clusterCoefficientUDF = udf(clusterCoefficient, FloatType())   

graph_features_df = graph_features_df.withColumn("cc", clusterCoefficientUDF(col("trian_count"), col("degree")))
graph_features_df = graph_features_df.fillna(0)
display(graph_features_df.orderBy(col("degree").desc()))

# COMMAND ----------

# DBTITLE 1,Community detection
communities = g.labelPropagation(maxIter=25)
display(communities)

# COMMAND ----------

# DBTITLE 1,Calculating community stats
comm_avg = communities.groupBy('label')\
                    .agg(F.avg("monthly_charges").alias("comm_avg_monthly_charges"), \
                         F.avg("total_charges").alias("comm_avg_total_charges"), \
                         F.avg("tenure").alias("comm_avg_tenure"), \
                         F.count("id").alias("comm_size")) 
display(comm_avg)

# COMMAND ----------

# DBTITLE 1,Deviation with average community values
communities = communities.join(comm_avg, on='label', how='left')
communities = communities.withColumn('comm_dev_avg_monthly_charges', F.col('comm_avg_monthly_charges')-F.col('monthly_charges'))
communities = communities.withColumn('comm_dev_avg_total_charges', F.col('comm_avg_total_charges')-F.col('total_charges'))
communities = communities.withColumn('comm_dev_avg_tenure', F.col('comm_avg_tenure')-F.col('tenure'))
display(communities)

# COMMAND ----------

graph_features_df = graph_features_df.alias('features')\
                 .join(communities.alias('comm'),\
                       communities.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'comm.comm_avg_monthly_charges', 'comm.comm_avg_total_charges', 'comm.comm_avg_tenure', 'comm.comm_size',\
                        'comm.comm_dev_avg_monthly_charges', 'comm.comm_dev_avg_total_charges', 'comm.comm_dev_avg_tenure') 
display(graph_features_df)

# COMMAND ----------

# DBTITLE 1,Calculating neighbour averages
edge_df_1 = edge_df.withColumnRenamed('src','id').withColumnRenamed('dst','nbgh')
edge_df_2 = edge_df.withColumnRenamed('dst','id').withColumnRenamed('src','nbgh')
und_edge_df = edge_df_1.union(edge_df_1)
und_edge_df = und_edge_df.alias('edge').join(vertex_df.select('id', 'monthly_charges', 'total_charges', 'tenure').alias('vertex'),\
                              und_edge_df.nbgh==vertex_df.id, how='left')\
                              .select('edge.*', 'vertex.monthly_charges', 'vertex.total_charges', 'vertex.tenure')\
                              .groupBy('id')\
                                  .agg(F.avg("monthly_charges").alias("nghb_avg_monthly_charges"), \
                                       F.avg("total_charges").alias("nghb_avg_total_charges"), \
                                       F.avg("tenure").alias("nghb_avg_tenure")) 
graph_features_df = graph_features_df.alias('features')\
                 .join(und_edge_df.alias('nbgh'),\
                       und_edge_df.id == graph_features_df.customer_id, 'left')\
                 .select('features.*', 'nbgh.nghb_avg_monthly_charges', 'nbgh.nghb_avg_total_charges', 'nbgh.nghb_avg_tenure') 
graph_features_df = graph_features_df.fillna(0)
display(graph_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Report

# COMMAND ----------

display(graph_features_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Write to Feature Store

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

try:
  #drop table if exists
  fs.drop_table(f'{dbName}.telco_churn_graph_features')
except:
  pass
#Note: You might need to delete the FS table using the UI
graph_feature_table = fs.create_table(
  name=f'{dbName}.telco_churn_graph_features',
  primary_keys='customer_id',
  schema=graph_features_df.schema,
  description='These features are derived from the telco customer call network.'
)

fs.write_table(df=graph_features_df, name=f'{dbName}.telco_churn_graph_features', mode='overwrite')

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Using Databricks AutoML to build our model
# MAGIC 
# MAGIC Next step: [Churn preiction model using AutoML]($./05_automl_churn_prediction)

# COMMAND ----------


