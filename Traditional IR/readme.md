# Traditional IR System for BioASQ Task 13b Phase A

This repository contains a BM25-based traditional information retrieval (IR) system implemented for the BioASQ challenge.

## 🔍 Overview

The system indexes a locally stored biomedical corpus and retrieves relevant documents for each test question using a classical BM25 model.

It was submitted to the BioASQ evaluation system and tested on the 2025 Task 13b Phase A dataset.

## 📁 Structure

- `generate_bioasq_ir_local.py` – Main script to index corpus and generate IR results
- `fetch_missing_pmids.py` – Utility to fetch abstracts using PubMed API
- `check_missing_pmids.py` – Compares local corpus vs. gold standard PMIDs
- `BioASQ-corpus.json` – Full corpus (abstracts + PMIDs)
- `results_traditionalIR_local.json` – Final submission file
- `screenshots/oracle-evaluation.png` – Evaluation results from the BioASQ Oracle tool

## 🛠️ How to Run

```bash
python generate_bioasq_ir_local.py
