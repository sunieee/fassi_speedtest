import faiss
import numpy as np

dim = 128
num_elements = 10000
num_queries = 100

# 整体上来说，要想要获得更快的构建和检索的速度，那么就需要把这三个超参相对地缩小，反之，要获得更好的召回精度，则需要将这三个超参增大。
M = 32 
efSearch = 100  # number of entry points (neighbors) we use on each layer
efConstruction = 100 # number of entry points used on each layer during construction

data = np.float32(np.random.random((num_elements, dim)))
ids = np.arange(num_elements)

queries = np.float32(np.random.random((num_queries,dim)))


# build hnsw
index = faiss.IndexHNSWFlat(dim, M)
index.hnsw.efConstruction = efConstruction
index.hnsw.efSearch = efSearch
index.add(data)

# query hnsw
top_k = 10
distance, preds = index.search(queries, k=top_k)

print(distance, preds)