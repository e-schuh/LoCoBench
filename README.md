# [Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias](https://arxiv.org/abs/2601.16934) - FairSentenceTransformers, Replication and Extension to Information Retrieval ![License: AGPLV3+](https://img.shields.io/badge/License-AGPLV3+-brightgreen.svg) 

---

## Overview

- **ACL 2026(findings)** — [*Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias*](https://aclanthology.org/2026.findings-acl.246/) - source code for the examination of positional bias in long documents of multilingual embedding models.
- **Preprint** — [*Attention Calibration for Position-Fair Dense Information Retrieval*](https://arxiv.org/abs/2606.02737) — extends the positional bias mitigation to downstream Information Retrieval Scenarios through the FairSentenceTransformers library.

---

## Table of Contents

- [Overview](#overview)
- [Fair Sentence Transformers](#fair-sentence-transformers)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Citation](#citation)
- [About Impresso](#about-impresso)
- [License](#license)

---

## Fair Sentence Transformers

We introduce an inference-time attention calibration method, implemented as an extension of Sentence Transformers called Fair Sentence Transformers. This tool aims to:

1. Provide a Wrapper Class for inference-time calibration techniques that improve fairness in embedding models.
2. Support existing and future embedding model releases through generic implementations configurable to each model's attributes.

# Datasets

We provide a multilingual comparable Wikipedia dataset for examining positional bias (see Section 3). The dataset includes six languages: English, Hindi, German, Italian, Korean, and Chinese.

Access the dataset: https://huggingface.co/datasets/impresso-project/wiki_comparable_corpus_en_de_hi_it_ko_zh
For instructions on loading and using the dataset, refer to the README on Hugging Face.

### Setup and Example use:

```bash
poetry install
```

```python
from src.fair_sentence_transformers.core.fair_sentence_transformer import FairSentenceTransformer
 
input_texts = [
    "What is the capital of Switzerland?",
    "How to make an omelette?",
    "Wie viele Einwohner hat Deutschland?",
]
 
model_name_or_path = "Alibaba-NLP/gte-multilingual-base"
model = FairSentenceTransformer(model_name_or_path)
 
# Standard SentenceTransformer embeddings
embeddings = model.encode(input_texts)  # shape: (3, 768)
 
# Fair SentenceTransformer embeddings
fair_embeddings = model.encode_positionally_fair(
    input_texts,
    calib_strength=0.5,
    calib_basket_size=128,
    calib_layers=6,
)  # shape: (3, 768)
```
pip install coming soon

### Supported Models and Methods

**encode_positionally_fair** - Inference-time attention calibration to ensure fair representation of input from all positions.

**Tested Models:**
- [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual), [107M version](ibm-granite/granite-embedding-107m-multilingual)
- Qwen3-Embedding Family: [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- microsoft/harrier-oss-v1: [0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b), 270M(WIP), 27B(WIP)

**Extensibility:** Our implementation can support additional models with a small configuration and quick test of new additions. Feel free put a pull request to add support to your favourite model or simply reach out to andrianos.michail@cl.uzh.ch

---

## Reproducing the Experiments

To fully replicate our results, please follow the instructions decipited in our Replication readme (REPL_README.md)

## Citation

If you use these resources, please cite our relevant works:

Positional and Language Bias, Attention Calibration (ACL2026):

```bibtex
@inproceedings{schuhmacher-etal-2026-information,
    title = "Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias",
    author = "Schuhmacher, Elias  and
      Michail, Andrianos  and
      Opitz, Juri  and
      Sennrich, Rico  and
      Clematide, Simon",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.246/",
    pages = "4996--5028",
    ISBN = "979-8-89176-395-1",
    abstract = "To be discoverable in an embedding-based search process, each part of a document should be reflected in its embedding representation. To quantify any potential reflection biases, we introduce a permutation-based evaluation framework. With this, we observe that state-of-the-art embedding models exhibit systematic positional and language biases when documents are longer and consist of multiple segments. Specifically, early segments and segments in higher-resource languages like English are over-represented, while later segments and segments in lower-resource languages are marginalized. In our further analysis, we find that the positional bias stems from front-loaded attention distributions in pooling-token embeddings, where early tokens receive more attention. To mitigate this issue, we introduce an inference-time attention calibration method that redistributes attention more evenly across document positions, increasing discoverabiltiy of later segments. Our evaluation framework and attention calibration is available at https://github.com/impresso/fair-sentence-transformers"
}
```

Attention Calibration for Information Retrieval (preprint):

```bibtex
@misc{michail2026attentioncalibrationpositionfairdense,
      title={Attention Calibration for Position-Fair Dense Information Retrieval}, 
      author={Andrianos Michail and Elias Schuhmacher and Juri Opitz and Simon Clematide and Rico Sennrich},
      year={2026},
      eprint={2606.02737},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2606.02737}, 
}
```

## About Impresso

### Impresso project

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders. The first project (2017-2021) was funded by the Swiss National Science Foundation under grant No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027) by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the Luxembourg National Research Fund under grant No. 17498891.

### Copyright

Copyright (C) 2026 The Impresso team.

### License

This program is provided as open source under the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE) v3 or later.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>
