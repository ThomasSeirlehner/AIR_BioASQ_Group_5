# Representation learning for BioASQ Task 13b Phase A
## 🔍 Overview

The system is based on the pre-trained sentence transformer "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb". The model is additionally fine-tuned with the BioASQ dataset. 

## 📁 Structure

- `fineTune.py` – is fine-tuning the base model with a Multiple-Negative-Ranking-Loss for document retrieval
- `fineTuneSnippet.py` – is fine-tuning the base model with Triplet-Loss for snippet retrieval
- `downloadDocuments.py` – downloads the documents from PubMed listed in the BioASQ dataset
- `downloadMissing.py` – download the missing documents
- `learn.py` – runs the test set with the fine-tuned model
- Other scripts were for testing. 

## 🛠️ How to Run
After cloning in git bash:
`python -m venv nlp_env`
`source nlp_env/Scripts/activate`
`pip install -r requirements.txt`

need to be installed: [Nvidia CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) 
