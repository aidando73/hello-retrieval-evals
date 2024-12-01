
# https://github.com/superlinked/VectorHub/blob/main/docs/articles/evaluation_of_RAG_retrieval_chunking_methods.md
# Model: https://huggingface.co/BAAI/bge-m3
# Dataset: squad
# k = 10
# chunk_size = 128

# Let's try:
# 1. Embed all contexts into chromadb
# 2. Query the questions against chromadb
# 3. Calculate the recall@k and precision@k by checking whether the chunk matches the document_id

from pprint import pprint
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=8192)["dense_vecs"]
embeddings_2 = model.encode(sentences_2, batch_size=12, max_length=8192)["dense_vecs"]

print(embeddings_1)
print(embeddings_2)

# Use sentence splitter to split the context into chunks

from llama_index.core.node_parser.text import SentenceSplitter
# from llama_index.core import VectorStoreIndex

text_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=0)

sentence3 = "In 2011, documents obtained by WikiLeaks revealed that Beyoncé was one of many entertainers who performed for the family of Libyan ruler Muammar Gaddafi. Rolling Stone reported that the music industry was urging them to return the money they earned for the concerts; a spokesperson for Beyoncé later confirmed to The Huffington Post that she donated the money to the Clinton Bush Haiti Fund. Later that year she became the first solo female artist to headline the main Pyramid stage at the 2011 Glastonbury Festival in over twenty years, and was named the highest-paid performer in the world per minute."
chunks = text_splitter.split_text(sentence3)

pprint(chunks)