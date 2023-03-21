# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph analytics for telecom customer churn prediction
# MAGIC 
# MAGIC Features extracted from telco customer network can provide valuable insights into the relationships and patterns of behavior among customers. Customer relationship data can be represented as a graph, where nodes represent customers and edges represent phone calls between customers.
# MAGIC 
# MAGIC By analyzing the call network graph features, machine learning models can identify patterns and predict which customers are most likely to churn. For example, machine learning models can analyze the network structure to identify customers who are more central or connected in the network, indicating that they may have a greater influence on other customers' behavior. Additionally, machine learning models can analyze the patterns of calls between customers, such as the frequency and duration of calls.
# MAGIC 
# MAGIC By combining these features with other customer data, such as demographics and usage patterns, machine learning models can build more accurate models for predicting customer churn. This can enable telecom companies to take proactive steps to retain customers and improve the customer experience, ultimately leading to increased customer loyalty and profitability.

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Graph analytics on Databricks
# MAGIC 
# MAGIC Graph analytics is a field of data analysis that focuses on extracting insights from data represented as graphs. A graph is a mathematical representation of a network of interconnected objects, where the objects are represented as nodes, and the connections between them are represented as edges.
# MAGIC 
# MAGIC In machine learning models, graph features are important because they can provide valuable insights into the relationships and connections between different entities. For example, in a social network, the connections between users can be represented as edges in a graph, and the features of each user can be represented as nodes. By analyzing the graph features, machine learning models can identify patterns and make predictions about user behavior.
# MAGIC 
# MAGIC Databricks provides a platform for advanced analytics, including graph analytics and machine learning. With Databricks, users can use Spark GraphFrames, a distributed graph processing system, to perform graph analytics on large datasets. Databricks also provides support for running machine learning algorithms on graph data, allowing users to train and deploy graph-based machine learning models. Additionally, Databricks provides a variety of tools and libraries for data visualization and exploration, making it easy to gain insights from graph data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## OLTP vs OLAP
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/nuwan-db/telco_churn_graph_analytics/blob/main/OLTP_vs_OLAP.png?raw=true" width="1000" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## Graph options with Databricks
# MAGIC 
# MAGIC ####GraphFrames
# MAGIC - DataFrame-based successor to Apache Spark GraphX
# MAGIC - E.g., AstraZeneca: [Building a Knowledge Graph with Spark and NLP: How We Recommend Novel Drugs to our Scientists](https://databricks.com/session_eu19/building-a-knowledge-graph-with-spark-and-nlp-how-we-recommend-novel-drugs-to-our-scientists)
# MAGIC 
# MAGIC ####DataFrames
# MAGIC - DIY: Many graph algorithms can be implemented via DataFrame joins.
# MAGIC - E.g., Paypal: [Optimize the Large Scale Graph Applications by using Apache Spark with 4-5x Performance Improvements](https://databricks.com/session_na20/optimize-the-large-scale-graph-applications-by-using-apache-spark-with-4-5x-performance-improvements)
# MAGIC 
# MAGIC ####3rd-party graph libraries
# MAGIC - E.g., NetworkX, Neo4J, TigerGraph, etc.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## GraphFrames user guide
# MAGIC 
# MAGIC GraphFrames is a package for Apache Spark that provides DataFrame-based graphs. It provides high-level APIs in Java, Python, and Scala. It aims to provide both the functionality of GraphX and extended functionality taking advantage of Spark DataFrames. This extended functionality includes motif finding, DataFrame-based serialization, and highly expressive graph queries.
# MAGIC 
# MAGIC Next: [Introduction to Apache Spark™ GraphFrames]($./01_graphframes_user_guide)

# COMMAND ----------


