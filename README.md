# Graph analytics for telecom customer churn prediction

Features extracted from telco customer network can provide valuable insights into the relationships and patterns of behavior among customers. Customer relationship data can be represented as a graph, where nodes represent customers and edges represent phone calls between customers.

By analyzing the call network graph features, machine learning models can identify patterns and predict which customers are most likely to churn. For example, machine learning models can analyze the network structure to identify customers who are more central or connected in the network, indicating that they may have a greater influence on other customers' behavior. Additionally, machine learning models can analyze the patterns of calls between customers, such as the frequency and duration of calls.

By combining these features with other customer data, such as demographics and usage patterns, machine learning models can build more accurate models for predicting customer churn. This can enable telecom companies to take proactive steps to retain customers and improve the customer experience, ultimately leading to increased customer loyalty and profitability.

## Graph analytics on Databricks

Graph analytics is a field of data analysis that focuses on extracting insights from data represented as graphs. A graph is a mathematical representation of a network of interconnected objects, where the objects are represented as nodes, and the connections between them are represented as edges.

In machine learning models, graph features are important because they can provide valuable insights into the relationships and connections between different entities. For example, in a social network, the connections between users can be represented as edges in a graph, and the features of each user can be represented as nodes. By analyzing the graph features, machine learning models can identify patterns and make predictions about user behavior.

Databricks provides a platform for advanced analytics, including graph analytics and machine learning. With Databricks, users can use Spark GraphFrames, a distributed graph processing system, to perform graph analytics on large datasets. Databricks also provides support for running machine learning algorithms on graph data, allowing users to train and deploy graph-based machine learning models. Additionally, Databricks provides a variety of tools and libraries for data visualization and exploration, making it easy to gain insights from graph data.

## OLTP vs OLAP

<img src="https://github.com/nuwan-db/Graph_Analytics_Telco_Churn_Prediction/blob/dev/_resources/images/OLTP_vs_OLAP.png?raw=true" width="1000"/>

## Graph options with Databricks

### GraphFrames
- DataFrame-based successor to Apache Spark GraphX
- E.g., AstraZeneca: [Building a Knowledge Graph with Spark and NLP: How We Recommend Novel Drugs to our Scientists](https://databricks.com/session_eu19/building-a-knowledge-graph-with-spark-and-nlp-how-we-recommend-novel-drugs-to-our-scientists)

### DataFrames
- DIY: Many graph algorithms can be implemented via DataFrame joins.
- E.g., Paypal: [Optimize the Large Scale Graph Applications by using Apache Spark with 4-5x Performance Improvements](https://databricks.com/session_na20/optimize-the-large-scale-graph-applications-by-using-apache-spark-with-4-5x-performance-improvements)

### 3rd-party graph libraries
- E.g., NetworkX, Neo4J, TigerGraph, etc.