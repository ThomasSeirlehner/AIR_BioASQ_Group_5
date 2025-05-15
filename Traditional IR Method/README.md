# Traditional IR Method for BioASQ Task

This script implements a traditional information retrieval approach for the BioASQ Challenge (Phase A).

## What it does

- Queries PubMed using the E-utilities API (ESearch + EFetch)
- Retrieves top 10 relevant documents for each question
- Extracts snippets from abstracts
- Outputs JSON formatted results compatible with BioASQ submission

## Files

- `generate_bioasq_traditional_ir.py`: Main script
- `BioASQ-task13bPhaseA-testset4.json`: Input test file (BIOASQ Challenge File)
- `results_traditionalIR_phaseA.json`: Example output (Generated file)

## How to run

```bash
pip install requests
python generate_bioasq_traditional_ir.py
