# Databricks notebook source
# MAGIC %md ## Introduction to Apache Spark™ GraphFrames
# MAGIC 
# MAGIC ####Origins
# MAGIC - Started in 2016 as a collaboration between Databricks, Berkeley, MIT
# MAGIC - Intended as a successor to Apache Spark GraphX
# MAGIC 
# MAGIC ####Current status
# MAGIC - Published as a [Spark Package](http://spark-packages.org/package/graphframes/graphframes)
# MAGIC - Supported by open-source contributors and committers
# MAGIC - Packaged with the Databricks Runtime for ML
# MAGIC 
# MAGIC This notebook demonstrates examples from the [GraphFrames User Guide](https://graphframes.github.io/graphframes/docs/_site/user-guide.html).

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import col, udf, lit, when
from graphframes import *
from pyspark.sql.types import *
from math import comb
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

# MAGIC %md ## Creating GraphFrames
# MAGIC 
# MAGIC Users can create GraphFrames from vertex and edge DataFrames.
# MAGIC 
# MAGIC * Vertex DataFrame: A vertex DataFrame should contain a special column named "id" which specifies unique IDs for each vertex in the graph.
# MAGIC * Edge DataFrame: An edge DataFrame should contain two special columns: "src" (source vertex ID of edge) and "dst" (destination vertex ID of edge).
# MAGIC 
# MAGIC Both DataFrames can have arbitrary other columns. Those columns can represent vertex and edge attributes.

# COMMAND ----------

# MAGIC %md Create the vertices first:

# COMMAND ----------

vertices = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)], ["id", "name", "age"])

# COMMAND ----------

# MAGIC %md And then some edges:

# COMMAND ----------

edges = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
  ("f", "c", "follow"),
  ("e", "f", "follow"),
  ("e", "d", "friend"),
  ("d", "a", "friend"),
  ("a", "e", "friend")
], ["src", "dst", "relationship"])

# COMMAND ----------

# MAGIC %md Let's create a graph from these vertices and these edges:

# COMMAND ----------

g = GraphFrame(vertices, edges)
draw_network(g, True)

# COMMAND ----------

# This example graph also comes with the GraphFrames package.
from graphframes.examples import Graphs
same_g = Graphs(sqlContext).friends()
draw_network(same_g, True)

# COMMAND ----------

# MAGIC %md ## Basic graph and DataFrame queries
# MAGIC 
# MAGIC GraphFrames provide several simple graph queries, such as node degree.
# MAGIC 
# MAGIC Also, since GraphFrames represent graphs as pairs of vertex and edge DataFrames, it is easy to make powerful queries directly on the vertex and edge DataFrames. Those DataFrames are made available as vertices and edges fields in the GraphFrame.

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

display(g.edges)

# COMMAND ----------

# MAGIC %md The incoming degree of the vertices:

# COMMAND ----------

display(g.inDegrees)

# COMMAND ----------

# MAGIC %md The outgoing degree of the vertices:

# COMMAND ----------

display(g.outDegrees)

# COMMAND ----------

# MAGIC %md The degree of the vertices:

# COMMAND ----------

display(g.degrees)

# COMMAND ----------

# MAGIC %md You can run queries directly on the vertices DataFrame. For example, we can find the age of the youngest person in the graph:

# COMMAND ----------

youngest = g.vertices.groupBy().min("age")
display(youngest)

# COMMAND ----------

# MAGIC %md Likewise, you can run queries on the edges DataFrame. For example, let's count the number of 'follow' relationships in the graph:

# COMMAND ----------

numFollows = g.edges.filter("relationship = 'follow'").count()
print("The number of follow edges is", numFollows)

# COMMAND ----------

# MAGIC %md ## Motif finding
# MAGIC 
# MAGIC Using motifs you can build more complex relationships involving edges and vertices. The following cell finds the pairs of vertices with edges in both directions between them. The result is a DataFrame, in which the column names are given by the motif keys.
# MAGIC 
# MAGIC Check out the [GraphFrame User Guide](https://graphframes.github.io/graphframes/docs/_site/user-guide.html#motif-finding) for more details on the API.

# COMMAND ----------

# Search for pairs of vertices with edges in both directions between them.
motifs = g.find("(a)-[e]->(b); (b)-[e2]->(a)")
display(motifs)

# COMMAND ----------

# MAGIC %md Since the result is a DataFrame, more complex queries can be built on top of the motif. Let us find all the reciprocal relationships in which one person is older than 30:

# COMMAND ----------

filtered = motifs.filter("b.age > 30 or a.age > 30")
display(filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stateful queries
# MAGIC 
# MAGIC Most motif queries are stateless and simple to express, as in the examples above. The next example demonstrates a more complex query that carries state along a path in the motif. Such queries can be expressed by combining GraphFrame motif finding with filters on the result where the filters use sequence operations to operate over DataFrame columns.
# MAGIC 
# MAGIC For example, suppose you want to identify a chain of 4 vertices with some property defined by a sequence of functions. That is, among chains of 4 vertices `a->b->c->d`, identify the subset of chains matching this complex filter:
# MAGIC 
# MAGIC * Initialize state on path.
# MAGIC * Update state based on vertex a.
# MAGIC * Update state based on vertex b.
# MAGIC * Etc. for c and d.
# MAGIC 
# MAGIC If final state matches some condition, then the filter accepts the chain.
# MAGIC The below code snippets demonstrate this process, where we identify chains of 4 vertices such that at least 2 of the 3 edges are “friend” relationships. In this example, the state is the current count of “friend” edges; in general, it could be any DataFrame Column.

# COMMAND ----------

# Find chains of 4 vertices.
chain4 = g.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d)")

# Query on sequence, with state (cnt)
#  (a) Define method for updating state given the next element of the motif.
def cumFriends(cnt, edge):
  relationship = col(edge)["relationship"]
  return when(relationship == "friend", cnt + 1).otherwise(cnt)

#  (b) Use sequence operation to apply method to sequence of elements in motif.
#   In this case, the elements are the 3 edges.
edges = ["ab", "bc", "cd"]
numFriends = reduce(cumFriends, edges, lit(0))
    
chainWith2Friends2 = chain4.withColumn("num_friends", numFriends).where(numFriends >= 2)
display(chainWith2Friends2)

# COMMAND ----------

# MAGIC %md ## Subgraphs
# MAGIC 
# MAGIC GraphFrames provides APIs for building subgraphs by filtering on edges and vertices. These filters can be composed together, for example the following subgraph only includes people who are more than 30 years old and have friends who are more than 30 years old.

# COMMAND ----------

g2 = g.filterEdges("relationship = 'friend'").filterVertices("age > 30").dropIsolatedVertices()

# COMMAND ----------

display(g2.vertices)

# COMMAND ----------

display(g2.edges)

# COMMAND ----------

draw_network(g2, True)

# COMMAND ----------

# MAGIC %md ## Standard graph algorithms
# MAGIC 
# MAGIC GraphFrames comes with a number of standard graph algorithms built in:
# MAGIC * Breadth-first search (BFS)
# MAGIC * Connected components
# MAGIC * Strongly connected components
# MAGIC * Label Propagation Algorithm (LPA)
# MAGIC * PageRank (regular and personalized)
# MAGIC * Shortest paths
# MAGIC * Triangle count

# COMMAND ----------

# MAGIC %md ###Breadth-first search (BFS)
# MAGIC 
# MAGIC Search from "Esther" for users of age < 32.

# COMMAND ----------

paths = g.bfs("name = 'Esther'", "age < 32")
display(paths)

# COMMAND ----------

# MAGIC %md The search may also be limited by edge filters and maximum path lengths.

# COMMAND ----------

filteredPaths = g.bfs(
  fromExpr = "name = 'Esther'",
  toExpr = "age < 32",
  edgeFilter = "relationship != 'friend'",
  maxPathLength = 3)
display(filteredPaths)

# COMMAND ----------

# MAGIC %md ## Connected components
# MAGIC 
# MAGIC Compute the connected component membership of each vertex and return a DataFrame with each vertex assigned a component ID. The GraphFrames connected components implementation can take advantage of checkpointing to improve performance.

# COMMAND ----------

sc.setCheckpointDir("/tmp/graphframes-example-connected-components")
result = g.connectedComponents()
display(result)

# COMMAND ----------

# MAGIC %md ## Strongly connected components
# MAGIC 
# MAGIC Compute the strongly connected component (SCC) of each vertex and return a DataFrame with each vertex assigned to the SCC containing that vertex.

# COMMAND ----------

result = g.stronglyConnectedComponents(maxIter=10)
display(result.select("id", "component"))

# COMMAND ----------

# MAGIC %md ## Label Propagation
# MAGIC 
# MAGIC Run static Label Propagation Algorithm for detecting communities in networks.
# MAGIC 
# MAGIC Each node in the network is initially assigned to its own community. At every superstep, nodes send their community affiliation to all neighbors and update their state to the most frequent community affiliation of incoming messages.
# MAGIC 
# MAGIC LPA is a standard community detection algorithm for graphs. It is very inexpensive computationally, although (1) convergence is not guaranteed and (2) one can end up with trivial solutions (all nodes are identified into a single community).

# COMMAND ----------

result = g.labelPropagation(maxIter=5)
display(result)

# COMMAND ----------

# MAGIC %md ## PageRank
# MAGIC 
# MAGIC PageRank is a measure of the importance or centrality of a node in a graph, originally developed by Larry Page and Sergey Brin while they were studying at Stanford University. It is used by the Google search engine to rank web pages in its search results.
# MAGIC 
# MAGIC The PageRank of a node in a graph is based on the idea that a node's importance is determined by the number and quality of the incoming links it receives from other nodes in the graph. In other words, the more incoming links a node has from other important nodes, the more important it is considered to be.
# MAGIC 
# MAGIC The PageRank algorithm assigns a score to each node in the graph based on this idea. The score of a node is calculated iteratively, by considering the scores of all the nodes that link to it, and the scores of all the nodes that those nodes link to, and so on. The algorithm uses a damping factor to prevent the score of a node from becoming too large, and it terminates after a fixed number of iterations or when the scores converge.
# MAGIC 
# MAGIC The PageRank score of a node can be used to rank the nodes in the graph by importance or centrality. Nodes with higher PageRank scores are considered to be more important or central to the graph. The PageRank algorithm is widely used in network analysis and information retrieval, and has been extended to many other applications beyond the web.
# MAGIC 
# MAGIC <img src="https://github.com/nuwan-db/telco_churn_graph_analytics/blob/main/pagerank.png?raw=true" width="600" />

# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices)

# COMMAND ----------

display(results.edges)

# COMMAND ----------

# Run PageRank for a fixed number of iterations.
g.pageRank(resetProbability=0.15, maxIter=10)

# COMMAND ----------

# Run PageRank personalized for vertex "a"
g.pageRank(resetProbability=0.15, maxIter=10, sourceId="a")

# COMMAND ----------

# MAGIC %md ## Shortest paths
# MAGIC 
# MAGIC Computes shortest paths to the given set of landmark vertices, where landmarks are specified by vertex ID.

# COMMAND ----------

results = g.shortestPaths(landmarks=["a", "d"])
display(results)

# COMMAND ----------

# MAGIC %md ###Triangle count
# MAGIC 
# MAGIC Computes the number of triangles passing through each vertex.

# COMMAND ----------

trian_count = g.triangleCount()
display(trian_count)

# COMMAND ----------

# DBTITLE 0,Defining custom graph matrices
# MAGIC %md ###Defining custom graph matrices
# MAGIC 
# MAGIC User defined function (UDFs) can be used to define custom graph matrices.

# COMMAND ----------

# MAGIC %md #### Clustering coefficient
# MAGIC 
# MAGIC The clustering coefficient of a node in a graph is a measure of the degree to which the neighbors of the node are connected to each other. It is defined as the ratio of the number of edges between the neighbors of the node to the maximum number of edges that could exist between them.
# MAGIC 
# MAGIC Clustering coefficient of a given node is defined as: 
# MAGIC $$ cc(i) = {\text{Number of complete triangles with coner } i \over \text{Number of all triangular graphs with coner } i} $$
# MAGIC 
# MAGIC Example:
# MAGIC 
# MAGIC <img src="https://github.com/nuwan-db/telco_churn_graph_analytics/blob/main/clustering_coefficient.png?raw=true" width="800" />

# COMMAND ----------

degree = g.degrees
custer_df = trian_count.join(degree, on='id', how='inner')

draw_network(g, False)

# COMMAND ----------

# Clustering coefficient

def clusterCoefficient(t, e):
  if e==0 or t==0:
    return 0.0
  else:
    return t/comb(e, 2)
  
clusterCoefficientUDF = udf(clusterCoefficient, FloatType())   

custer_df = custer_df.withColumn("cc", clusterCoefficientUDF(col("count"), col("degree")))

display(custer_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploratory Data Analysis
# MAGIC Our first job is to analyze and understand the data.
# MAGIC 
# MAGIC 
# MAGIC Next: [Exploratory Data Analysis]($./02_exploratory_data_analysis)

# COMMAND ----------


