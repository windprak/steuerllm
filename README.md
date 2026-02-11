# SteuerLLM: Local specialized large language model for German tax law analysis

## Overview

This repository contains the complete pipeline for developing a specialized large language model (LLM) trained on German tax legislation. The system encompasses data generation, model training, and evaluation components designed for legal domain adaptation.

## Repository Structure

### Data-Generation

**Pretraining-Filter**: Domain-specific filtering pipeline for identifying and extracting tax-related content from large-scale web corpora. Implements multi-worker architecture for efficient document classification and preprocessing.

**Instruct-Generation**: Implementation of the Water Fountain Algorithm for automated generation of instruction-tuning datasets. Comprises multiple generation strategies:
- Process-based parallelization for large-scale legislative text processing
- Thread-based execution for context-conditioned question-answer generation
- Specialized modules for accounting scenarios and legal commentary integration

The pipeline integrates semantic search (SearXNG), embedding-based retrieval, and LLM-driven synthesis to produce domain-specific question-answer pairs from German tax code (EStG, AO, KStG, etc.).

### Training

SLURM-based training configurations for distributed model training using Axolotl framework. Includes:
- Base model continual pretraining on filtered German tax corpus
- Instruction fine-tuning on generated QA datasets
- DeepSpeed ZeRO-3 optimization for multi-node H100 clusters
- FSDP2-compatible distributed training setup


## Methodology

The training procedure follows a two-stage approach: (1) continual pretraining on domain-filtered web data to adapt base model representations to tax-specific terminology and concepts, followed by (2) instruction fine-tuning on synthetically generated question-answer pairs derived from primary legal sources. The Water Fountain Algorithm employs retrieval-augmented generation with semantic ranking to ensure factual grounding and contextual relevance.

## Requirements

- Local SearXNG instance (port 8080) for semantic web retrieval
- Embedding server (port 8000) for similarity computation
- Tokenization server (port 8001) for context management
- LLM inference endpoints for QA generation
- SLURM cluster with GPU support for distributed training

## Citation

If you use this work, please cite:

```bibtex
@article{steuerllm,
  author = {Wind, Sebastian and Sopa, Jeta and Schmid, Laurin and Jackl, Quirin and Kiefer, Sebastian and Wu, Fei and Mayr, Martin and Köstler, Harald and Wellein, Gerhard and Maier, Andreas and Tayebi Arasteh, Soroosh},
  title = {SteuerLLM: Local specialized large language model for German tax law analysis},
  year = {2026},
  journal = {arxiv preprint}
}
```

## License

Research and academic use only.
