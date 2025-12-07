
## 👾 DIGIMON: Deep Analysis of Graph-Based Retrieval-Augmented Generation (RAG) Systems


<div style="text-align: center;">
  <a href="https://github.com/JayLZhou/GraphRAG"><img src="https://img.shields.io/badge/DIGIMON-red"/></a>
  <a href="https://github.com/JayLZhou/GraphRAG"><img src="https://img.shields.io/badge/Graph_RAG-red"/></a>
  <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/stars/JayLZhou/GraphRAG"/></a>
  <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/forks/JayLZhou/GraphRAG"/></a>
  <a href="http://makeapullrequest.com"><img src="https://img.shields.io/github/last-commit/JayLZhou/GraphRAG?color=blue"/></a>
</div>



<!-- ![Static Badge](https://img.shields.io/badge/DIGIMON-red)
![Static Badge](https://img.shields.io/badge/LLM-red)
![Static Badge](https://img.shields.io/badge/Graph_RAG-red)
![Static Badge](https://img.shields.io/badge/Document_QA-green)
![Static Badge](https://img.shields.io/badge/Document_Summarization-green) -->


<!-- <img src="img.png" alt="Description of the image" width="450" height="350"> -->

> **GraphRAG** is a popular 🔥🔥🔥 and powerful 💪💪💪 RAG system! 🚀💡 Inspired by systems like Microsoft's, graph-based RAG is unlocking endless possibilities in AI.

> Our project focuses on **modularizing and decoupling** these methods 🧩 to **unveil the mystery** 🕵️‍♂️🔍✨ behind them and share fun and valuable insights! 🤩💫  Our project🔨 is included in [Awesome Graph-based RAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

![Workflow of GraphRAG](workflow.png)

---
### Clone the repository

```bash
# Clone this repository (HTTPS)
git clone https://github.com/utkarshSinha1910/graphRagTxtToSql.git
cd graphRagTxtToSql
```

---
## Setup (Python + Ollama)

Use the commands below to install the exact Python version and Ollama used during development, then install the Python dependencies from `requirements.txt`. This is intentionally simple (no virtualenv steps).

- Install Python 3.9.6 (macOS) with `pyenv` (recommended to get an exact patch version):

```zsh
brew install pyenv
pyenv install 3.9.6
pyenv global 3.9.6
```

- Or install Python 3.9 via Homebrew (package name may vary):

```zsh
brew install python@3.9
```

- Install Python dependencies from `requirements.txt`:

```zsh
pip install --upgrade pip
pip install -r requirements.txt
```

- Install Ollama (macOS):

```zsh
brew install ollama
```

- Verify installations and versions:

```zsh
python3 --version   # should show 3.9.6 if installed
ollama --version    # recommended: 0.12.11
```

Notes:
- Ollama is a native CLI/tool (not a Python package). See https://ollama.com for downloads and docs.
- If you prefer not to use Ollama, set `OPENAI_API_KEY` and use the `OpenAIClient` in `Core/LLM/OpenAIClient.py`.

## Running locally (Ollama / phi3)

If you want to run models locally using Ollama, here are concise commands and examples for macOS (`zsh`). Replace `phi3` with the model you will use.

- Pull the model locally (if required):

```bash
ollama pull phi3
```

You can also check which models are already available locally and pull/verify `phi3` as follows:

```bash
# list locally available models
ollama ls

# pull the phi3 model from Ollama's model hub
ollama pull phi3

# verify the model is available locally
ollama ls | grep phi3
```

- Start the Ollama server (serves an HTTP API; default port commonly `11434`):

```bash
ollama serve
# runs in foreground; use nohup or & to background if needed
```
Run this on the another terminal
```bash
python3 main.py --task text2sql --spider_root Data/Spider --split dev --method gr
python3 main.py --task text2sql --spider_root Data/Spider --split dev --method dalk
python3 main.py --task text2sql --spider_root Data/Spider --split dev --method raptor
```

---



## Representative Methods

We select the following Graph RAG methods:

| Method | Description| Link | Graph Type|
| --- |--- |--- | :---: | 
| RAPTOR | ICLR 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2401.18059-b31b1b.svg)](https://arxiv.org/abs/2401.18059)  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/parthsarthi03/raptor)| Tree |
| KGP | AAAI 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2308.11730-b31b1b.svg)](https://arxiv.org/abs/2308.11730)  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YuWVandy/KG-LLM-MDQA)| Passage Graph |
| DALK | EMNLP 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.04819-b31b1b.svg)](https://arxiv.org/abs/2405.04819) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/David-Li0406/DALK)| KG |
| HippoRAG | NIPS 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.14831-b31b1b.svg)](https://arxiv.org/abs/2405.14831) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/OSU-NLP-Group/HippoRAG) | KG |
| G-retriever | NIPS 2024  | [![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)](https://arxiv.org/abs/2402.07630) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/XiaoxinHe/G-Retriever)| KG |
| ToG | ICLR 2024  | [![arXiv](https://img.shields.io/badge/arXiv-2307.07697-b31b1b.svg)](https://arxiv.org/abs/2307.07697) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/IDEA-FinAI/ToG)| KG |
| MS GraphRAG | Microsoft Project |  [![arXiv](https://img.shields.io/badge/arXiv-2404.16130-b31b1b.svg)](https://arxiv.org/abs/2404.16130) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/microsoft/graphrag)| TKG |
| FastGraphRAG | CircleMind Project  | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/circlemind-ai/fast-graphrag)| TKG |
| LightRAG | High Star Project  | [![arXiv](https://img.shields.io/badge/arXiv-2410.05779-b31b1b.svg)](https://arxiv.org/abs/2410.05779) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HKUDS/LightRAG)| RKG |


## Implemented Methods (in this repo)

This repository implements several concise heuristic graph builders (see `Core/Graph/GraphBuilder.py`). Below is a compact summary of the available builders and their key behaviors.

| Method | Short description | Graph Type | Key behavior |
|---|---:|:---:|---|
| `schema` | Schema-only graph (table → column) | Schema graph | Adds table and column nodes; table→column edges |
| `dalk` | Token-overlap + schema edges | KG / Passage | Adds schema edges and undirected semantic edges by token-overlap |
| `gr` | Graph-based retrieval (directed semantic links) | KG / Passage | Schema edges + directed semantic edges (overlap threshold) |
| `lgraphrag` | Local-search (top-K neighbors) | KG / Passage | Keeps top-k neighbors per node by overlap + schema edges |
| `ggraphrag` | Global-search (global similarity links) | KG / Passage | Connects globally similar nodes above threshold (bidirectional) |
| `hipporag` | PK-focused + intra-table links | KG | Emphasizes primary-key relationships and light semantic links |
| `kgp` | Co-occurrence / table-mention propagation | Passage Graph / KG | Adds co-occurrence edges and fallback semantic links |
| `lightrag` | Lightweight: schema + strong semantic matches | RKG | Only strong overlap edges plus schema edges (conservative) |
| `raptor` | PageRank-based pruning of semantic graph | Tree-like / Pruned Graph | Builds weighted semantic graph, runs PageRank and prunes weak edges |
| `tog` | Tree-of-Graphs backbone (MST) | Chunk Tree / KG backbone | Builds MST over semantic similarities and converts to bidirectional tree |

##  Graph Types
Based on the entity and relation, we categorize the graph into the following types:

+ **Chunk Tree**: A tree structure formed by document content and summary.
+ **Passage Graph**: A relational network composed of passages, tables, and other elements within documents.
+ **KG**: knowledge graph (KG) is constructed by extracting entities and relationships from each chunk, which contains only entities and relations, is commonly represented as triples.
+ **TKG**: A textual knowledge graph (TKG) is a specialized KG (following the same construction step as KG), which enriches entities with detailed descriptions and type information.
+ **RKG**: A rich knowledge graph (RKG), which further incorporates keywords associated with relations.

The criteria for the classification of graph types are as follows:

|Graph Attributes | Chunk Tree |Passage Graph | KG  | TKG | RKG |
| --- |--- |--- |--- | --- | --- |
|Original Content| ✅|✅| ❌|❌|❌| 
|Entity Name| ❌|❌|✅|✅|✅|
|Entity Type| ❌| ❌| ❌|✅|✅|
|Entity Description|❌| ❌| ❌|✅|✅|
|Relation Name| ❌|❌|✅|❌|✅|
|Relation keyword|❌| ❌| ❌|❌|✅|
|Relation Description|❌| ❌| ❌|✅|✅|
|Edge Weight| ❌|❌|✅|✅|✅|



## 🏹 Our future plans
- [ ] Detailed readme
- [ ] Support RoG, PathRAG, etc.
- [ ] Provide a docker image for easy deployment. 
- [ ] Support more LLMs, such as AZURE. 

## 🧭 Cite Our Paper

If you find this work useful, please consider citing our papers:

### In-depth Analysis of Graph-based RAG in a Unified Framework

```
@article{zhou2025depth,
  title={In-depth Analysis of Graph-based RAG in a Unified Framework},
  author={Zhou, Yingli and Su, Yaodong and Sun, Youran and Wang, Shu and Wang, Taotao and He, Runyuan and Zhang, Yongwei and Liang, Sicong and Liu, Xilin and Ma, Yuchi and others},
  journal={arXiv preprint arXiv:2503.04338},
  year={2025}
}
 ```


