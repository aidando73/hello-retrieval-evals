
# https://github.com/superlinked/VectorHub/blob/main/docs/articles/evaluation_of_RAG_retrieval_chunking_methods.md
# Model: https://huggingface.co/BAAI/bge-m3
# Dataset: squad
# k = 10

# Let's try:
# 1. Embed all contexts into chromadb
# 2. Query the questions against chromadb
# 3. Calculate the recall@k and precision@k by checking whether the chunk matches the document_id

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=8192)["dense_vecs"]
embeddings_2 = model.encode(sentences_2, batch_size=12, max_length=8192)["dense_vecs"]

print(embeddings_1)
print(embeddings_2)