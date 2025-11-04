# SEcurity-Feature-Framework-SEFF-

## Tech Stack: C, OpenMP, MPI, GCC, Rushmore HPC Cluster, Linux Shell
Predictive Vulnerability Detection: Using machine learning models to predict potential vulnerabilities in Programming Languages based on SEcurity Feature Framework (SEFF) criteria
SEFF is a valuable tool for identifying potential vulnerabilities in software code by assessing security features and patterns. However, the traditional approach of SEFF relies heavily on manual analysis, which can be
time-consuming and prone to human error. To address these limitations, we integrated machine learning techniques with SEFF to create a more automated and precise vulnerability detection mechanism. By combining SEFF's
structured analysis with the predictive power of machine learning, we aim to transform the way software vulnerabilities are identified and addressed, enhancing both speed and accuracy.



Engineered a parallel K-Means clustering system in C to perform large-scale customer segmentation using shared-memory (OpenMP) and distributed-memory (MPI) architectures.

Implemented a sequential baseline and progressively optimized versions with multi-threaded (OpenMP) and multi-process (MPI) parallelism to accelerate clustering across multiple nodes.

Leveraged OpenMP #pragma parallel for directives and synchronization mechanisms (atomic, critical) to safely parallelize centroid computation and data assignment loops.

Designed a distributed workload system using MPI_Scatterv, MPI_Reduce, and MPI_Gatherv for efficient communication and result aggregation across processes.

Evaluated the scalability of 2–16 processes on the Rushmore HPC cluster, achieving speedups up to 6× on medium datasets before network overhead impacted efficiency.

Integrated K-Means++ centroid initialization, data normalization, and dynamic workload partitioning, improving convergence stability and cluster quality.

Analyzed performance trade-offs between synchronization overhead and communication cost, and proposed a hybrid OpenMP + MPI model for future scalability.

Impact:
Demonstrated how parallel programming transforms computationally intensive algorithms into scalable, high-performance systems—reducing clustering time and enabling real-time segmentation for marketing and recommendation use cases.
