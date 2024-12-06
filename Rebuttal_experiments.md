# Section 1: Experimental Results in Dynamic Network Scenarios

## Node Failures
![Fig.1 Node Failure Probability vs. Deduplication Ratio](Node Error Rate vs Dedup.pdf "Fig.1 Node Failure Probability vs. Deduplication Ratio")

**Fig.1** The impact of node failure probability on the deduplication ratio.  
As the server failure rate in the system increases, the deduplication ratio of each method decreases to some extent. Among them, EdgeDup is least affected.

![Fig.2 Node Failure Probability vs. Retrieval Latency](Node Error Rate vs latency.pdf "Fig.2 Node Failure Probability vs. Retrieval Latency")

**Fig.2** The impact of node failure probability on retrieval latency.  
As the server failure rate increases, the retrieval latency for all methods rises. BEDD-O shows the most significant increase in latency, while both EdgeDup and MEAN exhibit relatively smaller increases.

![Fig.3 Index Maintenance Overhead due to Node Failures](Node Error vs communication.pdf "Fig.3 Index Maintenance Overhead due to Node Failures")

**Fig.3** The overhead of index maintenance caused by node failures.

## Addition of New Nodes
![Fig.4 Impact of New Node Addition on Retrieval Latency](Node Add vs latency.pdf "Fig.4 Impact of New Node Addition on Retrieval Latency")

**Fig.4** The effect of adding new nodes on retrieval latency.  
When new nodes join, the average retrieval latency decreases. This is because the newly added nodes can share data with their neighbors, reducing the number of data requests that need to be fetched from the cloud.

![Fig.5 Impact of New Node Addition on Deduplication Ratio](Node Add vs Dedup.pdf "Fig.5 Impact of New Node Addition on Deduplication Ratio")

**Fig.5** The effect of adding new nodes on the deduplication ratio.  
When new nodes are added, the deduplication ratio slightly decreases. This occurs because the introduction of new nodes brings in new data, and the LSS and LDS indexes may not have been updated yet, causing a temporary drop in deduplication ratio.

![Fig.6 Index Maintenance Overhead due to New Node Addition](Node Add vs communication.pdf "Fig.6 Index Maintenance Overhead due to New Node Addition")

**Fig.6** The index maintenance communication overhead incurred by adding new nodes.  
Compared to node failures, adding new nodes incurs greater index maintenance overhead.

## Network Disruptions Between Nodes
![Fig.7 Impact of Network Disruptions on Retrieval Latency](Node Error Rate vs latency.pdf "Fig.7 Impact of Network Disruptions on Retrieval Latency")

**Fig.7** The effect of network disruptions between nodes on retrieval latency.  
When network disruptions occur between servers, the retrieval latency increases for every method as the number of disruptions grows. BEDD-O’s latency increases most prominently, while EdgeDup and MEAN experience relatively mild increases.

![Fig.8 Impact of Network Disruptions on Deduplication Ratio](Node Error Rate vs Dedup.pdf "Fig.8 Impact of Network Disruptions on Deduplication Ratio")

**Fig.8** The effect of network disruptions on the deduplication ratio.  
Network failures have a relatively minor impact on the overall deduplication ratio.

# Section 2: Decision and Execution Time
![Fig.9 Decision Time of Distributed vs. Centralized Deduplication Methods](Server Num vs decision.pdf "Fig.9 Decision Time of Distributed vs. Centralized Methods")

**Fig.9** Decision time comparison.

![Fig.10 Execution Time of Distributed vs. Centralized Deduplication Methods](Server Num vs execute.pdf "Fig.10 Execution Time of Distributed vs. Centralized Methods")

**Fig.10** Execution time comparison.  
In terms of decision time, EdgeDup performs distributed deduplication decisions locally, and this decision time does not increase as the number of edge servers grows. During execution, due to the low-latency edge environment, EdgeDup’s execution time is significantly lower than that of centralized methods.

![Fig.11 Total Deduplication Time of Distributed vs. Centralized Deduplication Methods](Server Num vs Deduplication Time.pdf "Fig.11 Total Deduplication Time")

**Fig.11** Total deduplication time comparison.  
With decision time comparable to MEAN and execution time lower than MEAN, EdgeDup achieves the lowest total deduplication completion time.

# Section 3: The Impact of the Alpha Parameter
![Fig.12 Alpha Parameter vs. Retrieval Latency](alpha vs latency.pdf "Fig.12 Alpha Parameter vs. Retrieval Latency")

**Fig.12** The effect of the alpha parameter on retrieval latency.

![Fig.13 Alpha Parameter vs. Deduplication Ratio](alpha vs max_dedup.pdf "Fig.13 Alpha Parameter vs. Deduplication Ratio")

**Fig.13** The effect of the alpha parameter on deduplication ratio.  
“Optimal” represents the centralized approach, while “EdgeDup” represents the distributed optimization. As the alpha value increases, greater emphasis is placed on deduplication ratio, resulting in higher deduplication ratio but also higher retrieval latency.

# Section 4: Index Storage Overhead
![Fig.14 Network Density vs. Average LSS and LDS Storage Space](density vs memory.pdf "Fig.14 Network Density vs. Average LSS and LDS Storage Space")

**Fig.14** The impact of network density on the average LSS and LDS storage space per data item.  
As network density increases, the average LSS and LDS storage space per data item initially increases and then stabilizes. This occurs because when the network density exceeds 0.28, the number of neighbors for each node tends to approach the total number of servers.
