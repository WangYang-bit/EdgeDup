# Section 1: Experimental Results in Dynamic Network Scenarios

## Node Failures
![Fig.1 Node Failure Probability vs. Deduplication Ratio](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Node%20Error%20Rate%20vs%20Dedup.png "Fig.1 Node Failure Probability vs. Deduplication Ratio")

**Fig.1** The impact of node failure probability on the deduplication ratio.  
As the server failure rate in the system increases, the deduplication ratio of each method decreases to some extent. Among them, EdgeDup is least affected.

![Fig.2 Node Failure Probability vs. Retrieval Latency](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Node%20Error%20Rate%20vs%20latency.png "Fig.2 Node Failure Probability vs. Retrieval Latency")

**Fig.2** The impact of node failure probability on retrieval latency.  
As the server failure rate increases, the retrieval latency for all methods rises. BEDD-O shows the most significant increase in latency, while both EdgeDup and MEAN exhibit relatively smaller increases.

![Fig.3 Index Maintenance Overhead due to Node Failures](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Node%20Error%20vs%20communication.png "Fig.3 Index Maintenance Overhead due to Node Failures")

**Fig.3** The overhead of index maintenance caused by node failures.

## Addition of New Nodes
![Fig.4 Impact of New Node Addition on Retrieval Latency](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Node%20Add%20vs%20latency.png "Fig.4 Impact of New Node Addition on Retrieval Latency")

**Fig.4** The effect of adding new nodes on retrieval latency.  
When new nodes join, the average retrieval latency decreases. This is because the newly added nodes can share data with their neighbors, reducing the number of data requests that need to be fetched from the cloud.

![Fig.5 Impact of New Node Addition on Deduplication Ratio](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Node%20Add%20vs%20Dedup.png "Fig.5 Impact of New Node Addition on Deduplication Ratio")

**Fig.5** The effect of adding new nodes on the deduplication ratio.  
When new nodes are added, the deduplication ratio slightly decreases. This occurs because the introduction of new nodes brings in new data, and the LSS and LDS indexes may not have been updated yet, causing a temporary drop in deduplication ratio.

![Fig.6 Index Maintenance Overhead due to New Node Addition](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Node%20Add%20vs%20communication.png "Fig.6 Index Maintenance Overhead due to New Node Addition")

**Fig.6** The index maintenance communication overhead incurred by adding new nodes.  
Compared to node failures, adding new nodes incurs greater index maintenance overhead.

## Network Disruptions Between Nodes
![Fig.7 Impact of Network Disruptions on Retrieval Latency](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Node%20Error%20Rate%20vs%20latency.png "Fig.7 Impact of Network Disruptions on Retrieval Latency")

**Fig.7** The effect of network disruptions between nodes on retrieval latency.  
When network disruptions occur between servers, the retrieval latency increases for every method as the number of disruptions grows. BEDD-O’s latency increases most prominently, while EdgeDup and MEAN experience relatively mild increases.

![Fig.8 Impact of Network Disruptions on Deduplication Ratio](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Node%20Error%20Rate%20vs%20Dedup.png "Fig.8 Impact of Network Disruptions on Deduplication Ratio")

**Fig.8** The effect of network disruptions on the deduplication ratio.  
Network failures have a relatively minor impact on the overall deduplication ratio.

# Section 2: Decision and Execution Time
![Fig.9 Decision Time of Distributed vs. Centralized Deduplication Methods](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Server%20Num%20vs%20decision.png "Fig.9 Decision Time of Distributed vs. Centralized Methods")

**Fig.9** Decision time comparison.

![Fig.10 Execution Time of Distributed vs. Centralized Deduplication Methods](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Server%20Num%20vs%20execute.png "Fig.10 Execution Time of Distributed vs. Centralized Methods")

**Fig.10** Execution time comparison.  
In terms of decision time, EdgeDup performs distributed deduplication decisions locally, and this decision time does not increase as the number of edge servers grows. During execution, due to the low-latency edge environment, EdgeDup’s execution time is significantly lower than that of centralized methods.

![Fig.11 Total Deduplication Time of Distributed vs. Centralized Deduplication Methods](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Server%20Num%20vs%20Deduplication%20Time.png "Fig.11 Total Deduplication Time")

**Fig.11** Total deduplication time comparison.  
With decision time comparable to MEAN and execution time lower than MEAN, EdgeDup achieves the lowest total deduplication completion time.

# Section 3: The Impact of the Alpha Parameter
![Fig.12 Alpha Parameter vs. Retrieval Latency](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/alpha%20vs%20latency.png "Fig.12 Alpha Parameter vs. Retrieval Latency")

**Fig.12** The effect of the alpha parameter on retrieval latency.

![Fig.13 Alpha Parameter vs. Deduplication Ratio](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/alpha%20vs%20max_dedup.png "Fig.13 Alpha Parameter vs. Deduplication Ratio")

**Fig.13** The effect of the alpha parameter on deduplication ratio.  
“Optimal” represents the centralized approach, while “EdgeDup” represents the distributed optimization. As the alpha value increases, greater emphasis is placed on deduplication ratio, resulting in higher deduplication ratio but also higher retrieval latency.

# Section 4: Index Storage Overhead
![Fig.14 Network Density vs. Average LSS and LDS Storage Space](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/density%20vs%20memory.png "Fig.14 Network Density vs. Average LSS and LDS Storage Space")

**Fig.14** The impact of network density on the average LSS and LDS storage space per data item.  
As network density increases, the average LSS and LDS storage space per data item initially increases and then stabilizes. This occurs because when the network density exceeds 0.28, the number of neighbors for each node tends to approach the total number of servers.

# Section 5: The Impact of Popularity Prediction Bias
![Fig.15 Impact of Popularity Prediction Bias on Retrieval Latency](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Popularity%20Prediction%20Bias%20vs%20Latency.png "Fig.15 Impact of Popularity Prediction Bias on Retrieval Latency")

**Fig.15** The impact of popularity prediction bias on retrieval latency.

![Fig.16 Impact of Popularity Prediction Bias on Deduplication Ratio](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Popularity%20Prediction%20Bias%20vs%20Dedup.png "Fig.16 Impact of Popularity Prediction Bias on Deduplication Ratio")

**Fig.16** The impact of popularity prediction bias on the deduplication ratio.

# Section 6: Communication Overhead Analysis between Centralized and Distributed Methods
![Fig.17 Communication Overhead between Centralized and Distributed Methods](https://github.com/WangYang-bit/EdgeDup/blob/main/Figures/Centralized%20vs%20Distributed%20Communication%20Overhead.png "Fig.17 Communication Overhead between Centralized and Distributed Methods")

**Fig.17** Communication overhead comparison between centralized and distributed methods.

# Section 7: The Impact of Network Density on EdgeDup

| Network Density | Data Retrieval Latency (ms) | Deduplication Ratio (\%) | Communication Overhead (Messages) | 
| --------------- | --------------------------- | ------------------------ | ---------------------------------- | 
| 0.3             | 16.04                       | 59.6                     | 782                                | 
| 0.4             | 12.96                       | 62.1                     | 1885                               | 
| 0.5             | 11.72                       | 66.1                     | 2435                               | 
| 0.6             | 11.42                       | 68.6                     | 2604                               | 
| 0.7             | 11.07                       | 68.5                     | 2634                               | 

**Table 1**: Impact of Network Density on EdgeDup Performance Metrics.
