
```bash
pip install -r requirements.txt

source ~/miniconda3/bin/activate
conda create --prefix ./env python=3.10

conda activate ./env
pip install -r requirements.txt
# Ran into issues with pytorch faiss
conda install -c pytorch faiss-cpu=1.9.0
conda install -c conda-forge faiss-cpu
```


Differences between squad_2.ipynb and chunker-research:
- Docs are de-duplicated and given an id
- A match => if the chunk originates from the same doc (not whether it contains the answer as substring)
- Uses Exact search via Inner Product search instead of HNSW via L2 distance

Other minor differences
- Chunk overlap is 16
- train and validation splits are combined

Thank you :thanks: I was finding a way to evaluate retrieval results for llama-stack so I could improve retrievals - so thanks for making this open source.