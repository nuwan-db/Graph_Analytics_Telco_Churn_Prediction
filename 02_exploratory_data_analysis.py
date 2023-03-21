# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false

# COMMAND ----------

import plotly.express as px
from graphframes import *
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F
import networkx as nx

# COMMAND ----------

# A support fucntion for drawing networks

def draw_network(g, directed=False, with_labels=True, node_color='#F39C12'):
  if directed:
    nxg = nx.DiGraph()
  else:
    nxg = nx.Graph()
    
  nx.draw(nx.from_pandas_edgelist(g.edges.toPandas(), create_using=nxg, source='src', target='dst'), \
          arrows=directed, with_labels=with_labels, node_color=node_color, \
          connectionstyle='arc3, rad = 0.1')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Data
# MAGIC 
# MAGIC The data is provided by Home Credit, a service dedicated to provided lines of credit (loans) to the unbanked population. Predicting whether or not a client will repay a loan or have difficulty is a critical business need, and Home Credit is hosting this competition on Kaggle to see what sort of models the machine learning community can develop to help them in this task.
# MAGIC 
# MAGIC There are different sources of data:
# MAGIC 
# MAGIC - **application_train/application_test**: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid.
# MAGIC - **bureau**: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.

# COMMAND ----------

# Read customer data into Spark
customer_df = spark.table("telco_churn_customers_bronze")
display(customer_df)

# COMMAND ----------

# Read call log data into Spark
call_df = spark.table("telco_call_log_bronze")
display(call_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Profiling

# COMMAND ----------

display(customer_df)

# COMMAND ----------

display(call_df)

# COMMAND ----------

# MAGIC %md
# MAGIC As you may have noticed, the data type for some of the fields are wrong. As example, the data type of total_charges should be changed to double and the data type call datetime should be changed to datetime. 

# COMMAND ----------

customer_df = customer_df.withColumn("total_charges", customer_df["total_charges"].cast(DoubleType()))

# COMMAND ----------

call_df = call_df.withColumn("datatime", F.to_timestamp("datatime"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Visualization Plotly

# COMMAND ----------

fig = px.histogram(customer_df.toPandas().sample(n=10000, replace=True, random_state=123).sort_index(), x='payment_method', color='churn')
fig.update_xaxes(categoryorder='total descending')
fig.show()

# COMMAND ----------

fig = px.pie(df, names='contract')
fig.show()

# COMMAND ----------

fig = px.histogram(df, x="tenure", color="churn", 
             color_discrete_map={'No':'orange',
                                 'Yes':'green'})
fig.show()

# COMMAND ----------

# MAGIC %md ## Graph visualizations
# MAGIC 
# MAGIC Graph visualizations are important for understanding the properties of a graph because they can reveal patterns and relationships that may not be immediately obvious from the raw data. Here are some reasons why graph visualizations are helpful:
# MAGIC 
# MAGIC - Identification of structure: Graph visualizations can help identify the structure of the graph, such as whether it is a tree, a chain, a star, or a more complex network. The visual representation can also reveal patterns, clusters, and outliers in the data.
# MAGIC 
# MAGIC - Clarity and communication: Graph visualizations can be used to communicate complex relationships in a clear and concise way. They can help users understand the data more quickly and easily than with a table or a list of numbers.
# MAGIC 
# MAGIC - Exploration and analysis: Graph visualizations can be used to explore the data and analyze its properties. For example, they can be used to identify nodes with high degrees, identify cliques or communities, or visualize the shortest path between two nodes.
# MAGIC 
# MAGIC - Visualization of dynamic data: Graph visualizations can be used to represent dynamic data, such as changes in a social network over time or changes in a supply chain. These visualizations can help users understand how the data changes over time and identify trends and patterns.
# MAGIC 
# MAGIC Overall, graph visualizations can provide a powerful tool for understanding the properties of a graph and the relationships within it. By providing a visual representation of the data, they can help users identify patterns, explore and analyze the data, and communicate their findings to others.Many third party vizualization libraries are available for graph visualizations within Databricks notebooks.
# MAGIC 
# MAGIC 
# MAGIC Example: A visualization of 20% of the telco customer network using [Graphistry Python library](https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.ipynb):
# MAGIC <img src="https://github.com/nuwan-db/telco_churn_graph_analytics/blob/main/Telco_network_viz_20pcn.png?raw=true" width="1000" />

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graph Data Analysis
# MAGIC 
# MAGIC We need to prepare vertex and edge DataFrames to create a GraphFrame for customer call network.
# MAGIC 
# MAGIC - Vertex DataFrame: A vertex DataFrame should contain a special column named "id" which specifies unique IDs for each vertex in the graph.
# MAGIC - Edge DataFrame: An edge DataFrame should contain two special columns: "src" (source vertex ID of edge) and "dst" (destination vertex ID of edge).

# COMMAND ----------

# MAGIC %md
# MAGIC Vertex DataFrame can be created using the customer_df.

# COMMAND ----------

vertex_df = customer_df.withColumnRenamed('customer_id', 'id')
display(vertex_df)

# COMMAND ----------

# MAGIC %md
# MAGIC To create an edge DataFrame from the call logs, we first need to map mobile_number to customer_id. 

# COMMAND ----------

edge_df = call_df.alias('call').join(customer_df.alias('customer'), call_df.caller_mobile_number==customer_df.mobile_number, how='left')\
                 .select('customer.customer_id','call.callee_mobile_number')\
                 .withColumnRenamed('customer_id','src')
edge_df = edge_df.alias('call').join(customer_df.alias('customer'), call_df.callee_mobile_number==customer_df.mobile_number, how='left')\
                 .select('customer.customer_id','call.src')\
                 .withColumnRenamed('customer_id','dst')
edge_df = edge_df.dropDuplicates()

# COMMAND ----------

display(edge_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a graph from these vertices and these edges:

# COMMAND ----------

g = GraphFrame(vertex_df, edge_df)

# COMMAND ----------

#The degree of the vertices
display(g.degrees.orderBy(F.desc("degree")))

# COMMAND ----------

# Detecting connected components to understand the network connectivity:
sc.setCheckpointDir("/tmp/graphframes-example-connected-components")
result = g.connectedComponents()
display(result)

# COMMAND ----------

# Understanding the community structures in the network:
result = g.labelPropagation(maxIter=5)
display(result)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Saving updated dataframes

# COMMAND ----------

customer_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("telco_churn_customers_silver")
call_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("telco_call_log_silver")
vertex_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("telco_vertex_df")
edge_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("telco_edge_df")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Feature Engineering
# MAGIC Our next job is to  prepare a set of features that we'll be able to use in customer churn prediction and other data science projects.
# MAGIC 
# MAGIC 
# MAGIC Next: [Feature engineering]($./03_customer_feature_engineering)

# COMMAND ----------


