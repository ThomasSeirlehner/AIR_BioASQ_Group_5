# BioASQ Task 3 – Neural Re-ranking

This repository contains code and data for BioASQ Task 3 (13b) – Neural Re-ranking of snippets using a fine-tuned BioBERT model.

## Contents

- `JSON_Formatting.py`: Script to convert raw model outputs into BioASQ submission format.
- `rerank_bioasq.py`: Main script that loads the fine-tuned model and performs re-ranking.
- `results_traditionalIR_phaseA.json`: Initial retrieval results from the traditional IR model.
- `Task_3_reranked_results.json`: Final output in BioASQ submission format after reranking.
- `training13b.json`: The training file used for model fine-tuning.
- `requirements.txt`: Python dependencies.

## How to Use

1. Create a virtual environment and activate it:
   ```bash
   python -m venv bioasq_venv
   source bioasq_venv/bin/activate  # or bioasq_venv\Scripts\activate on Windows

2. Install dependencies:

pip install -r requirements.txt


3. Run the re-ranking pipeline:

python rerank_bioasq.py


4. Format the output for submission:

python JSON_Formatting.py
