
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
from typing import List
from FlagEmbedding import BGEM3FlagModel
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding
import numpy as np

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=8192)["dense_vecs"]
embeddings_2 = model.encode(sentences_2, batch_size=12, max_length=8192)["dense_vecs"]

# print(embeddings_1)
# print(embeddings_2)

class BGE_M3_Embedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return model.encode(query, batch_size=12, max_length=8192)["dense_vecs"]

    def _get_text_embedding(self, text: str) -> List[float]:
        return model.encode(text, batch_size=12, max_length=8192)["dense_vecs"]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return model.encode(texts, batch_size=12, max_length=8192)["dense_vecs"]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

embedding_model = BGE_M3_Embedding()


print(embedding_model._get_query_embedding(sentences_1[0]))
print(embeddings_1[0])
assert np.array_equal(embedding_model._get_query_embedding(sentences_1[0]), model.encode(sentences_1[0], batch_size=12, max_length=8192)["dense_vecs"])
# assert np.array_equal(embedding_model._get_query_embedding(sentences_1[1]), embeddings_1[1])
assert np.array_equal(embedding_model._get_text_embedding(sentences_2[0]), model.encode(sentences_2[0], batch_size=12, max_length=8192)["dense_vecs"])
# assert np.array_equal(embedding_model._get_text_embedding(sentences_2[1]), embeddings_2[1])
assert np.array_equal(embedding_model._get_text_embeddings(sentences_2), embeddings_2)

# Use sentence splitter to split the context into chunks

from llama_index.core.node_parser.text import SentenceSplitter
# from llama_index.core import Document
# from llama_index.core.ingestion import IngestionPipeline
# from llama_index.core import VectorStoreIndex

text_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=0)

sentence3 = "In 2011, documents obtained by WikiLeaks revealed that Beyoncé was one of many entertainers who performed for the family of Libyan ruler Muammar Gaddafi. Rolling Stone reported that the music industry was urging them to return the money they earned for the concerts; a spokesperson for Beyoncé later confirmed to The Huffington Post that she donated the money to the Clinton Bush Haiti Fund. Later that year she became the first solo female artist to headline the main Pyramid stage at the 2011 Glastonbury Festival in over twenty years, and was named the highest-paid performer in the world per minute."
sentence4 = "The sky is usually clear above the desert and the sunshine duration is extremely high everywhere in the Sahara. Most of the desert enjoys more than 3,600 h of bright sunshine annually or over 82% of the time and a wide area in the eastern part experiences in excess of 4,000 h of bright sunshine a year or over 91% of the time, and the highest values are very close to the theoretical maximum value. A value of 4,300 h or 98% of the time would be recorded in Upper Egypt (Aswan, Luxor) and in the Nubian Desert (Wadi Halfa). The annual average direct solar irradiation is around 2,800 kWh/(m2 year) in the Great Desert. The Sahara has a huge potential for solar energy production. The constantly high position of the sun, the extremely low relative humidity, the lack of vegetation and rainfall make the Great Desert the hottest continuously large area worldwide and certainly the hottest place on Earth during summertime in some spots. The average high temperature exceeds 38 °C (100.4 °F) - 40 °C (104 °F) during the hottest month nearly everywhere in the desert except at very high mountainous areas. The highest officially recorded average high temperature was 47 °C (116.6 °F) in a remote desert town in the Algerian Desert called Bou Bernous with an elevation of 378 meters above sea level. It's the world's highest recorded average high temperature and only Death Valley, California rivals it. Other hot spots in Algeria such as Adrar, Timimoun, In Salah, Ouallene, Aoulef, Reggane with an elevation between 200 and 400 meters above sea level get slightly lower summer average highs around 46 °C (114.8 °F) during the hottest months of the year. Salah, well known in Algeria for its extreme heat, has an average high temperature of 43.8 °C (110.8 °F), 46.4 °C (115.5 °F), 45.5 (113.9 °F). Furthermore, 41.9 °C (107.4 °F) in June, July, August and September. In fact, there are even hotter spots in the Sahara, but they are located in extremely remote areas, especially in the Azalai, lying in northern Mali. The major part of the desert experiences around 3 – 5 months when the average high strictly exceeds 40 °C (104 °F). The southern central part of the desert experiences up to 6 – 7 months when the average high temperature strictly exceeds 40 °C (104 °F) which shows the constancy and the length of the really hot season in the Sahara. Some examples of this are Bilma, Niger and Faya-Largeau, Chad. The annual average daily temperature exceeds 20 °C (68 °F) everywhere and can approach 30 °C (86 °F) in the hottest regions year-round. However, most of the desert has a value in excess of 25 °C (77 °F). The sand and ground temperatures are even more extreme. During daytime, the sand temperature is extremely high as it can easily reach 80 °C (176 °F) or more. A sand temperature of 83.5 °C (182.3 °F) has been recorded in Port Sudan. Ground temperatures of 72 °C (161.6 °F) have been recorded in the Adrar of Mauritania and a value of 75 °C (167 °F) has been measured in Borkou, northern Chad. Due to lack of cloud cover and very low humidity, the desert usually features high diurnal temperature variations between days and nights. However, it's a myth that the nights are cold after extremely hot days in the Sahara. The average diurnal temperature range is typically between 13 °C (55.4 °F) and 20 °C (68 °F). The lowest values are found along the coastal regions due to high humidity and are often even lower than 10 °C (50 °F), while the highest values are found in inland desert areas where the humidity is the lowest, mainly in the southern Sahara. Still, it's true that winter nights can be cold as it can drop to the freezing point and even below, especially in high-elevation areas."
chunks = text_splitter.split_text(sentence4)

# document = Document(text=sentence4)

# pprint(chunks)
# pprint(document)


# pipeline = IngestionPipeline(
#     transformations=[
#         text_splitter,
#         embedding_model
#     ]
# )

# nodes = pipeline.run(documents=[document])

# Not going to use VectorStoreIndex for now - fairly large abstraction
# index = VectorStoreIndex(nodes=nodes)

# pprint(nodes)

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm

chroma_client = chromadb.Client()




from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        text_split = [text_splitter.split_text(text) for text in input]
        embeddings = embedding_model._get_text_embeddings(text_split)
        return embeddings

my_embedding_function = MyEmbeddingFunction()

collection = chroma_client.create_collection(name="my_collection", embedding_function=my_embedding_function)

import datasets

# https://huggingface.co/datasets/hotpotqa/hotpot_qa?row=16
# https://arxiv.org/pdf/1606.05250
dataset = datasets.load_dataset("rajpurkar/squad")

# Pretty slow
for i in tqdm(range(len(dataset["train"]))):
    row = dataset["train"][i]
    chunks = text_splitter.split_text(row["context"])
    embeddings = embedding_model._get_text_embeddings(chunks)
    collection.add(embeddings=embeddings, documents=chunks, ids=[f"{row['id']}_{i}" for i in range(len(chunks))])

collection.add(
    ids=[row["id"] for row in dataset["train"]],
    documents=[row["context"] for row in dataset["train"]]
)

