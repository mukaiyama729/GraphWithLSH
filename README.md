# GraphWithLSH
This is an algorithm using LSH graph.  The created graph is an approximate knn graph.
Rust code is utilized on back side during calculations of kernel functions and distance matrix.

# Features
- Up to around 100,000 points is within the acceptable range.
- Calculations for around 100,000 points can be completed in about 2 minutes.
# Reference

[Approximate knn](https://graphics.stanford.edu/courses/cs468-06-fall/Papers/06%20indyk%20motwani%20-%20stoc98.pdf)

[LSH kNN graph for diffusion on image retrieval](https://link.springer.com/article/10.1007/s10791-020-09388-8)

[Local sesitivity hashing](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)


