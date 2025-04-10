# list

## Requirements
```bash
conda create -n bioasq_list python=3.7
conda activate bioasq_list
pip install tensorflow-gpu==1.14.0
pip install tqdm
```

## Fine-tuning
Please use `run_list.py` for list type questions. Use `BioASQ-*.json` for training and predicting dataset.
Our pre-processed version of the BioASQ Task 8b-Phase B dataset is available at **[`Pubmed Abstract`](https://drive.google.com/drive/folders/1JPMC4P7dgeoG-JW3yMKY8t6bnym5-GNb?usp=sharing)** and **[`List type`](https://drive.google.com/file/d/1qpUvMosQ8ufIfuyeyKcVlsyEa7J0H7PY/view?usp=sharing)**.
Follow the below processes to train and predict for our model.

### Training and Predicting
If you want to predict then set the `--do_predict` argument as True else False.

```bash
export BIOBERT_DIR="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/biobert_v1.1_pubmed"
export DATA_DIR="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/BioASQ-training8b"
export batch=1

#python run_list.py --do_train=True --do_predict=True --vocab_file="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/biobert_v1.1_pubmed/vocab.txt" --bert_config_file="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/biobert_v1.1_pubmed/bert_config.json" --init_checkpoint="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/biobert_v1.1_pubmed/model.ckpt-1000000" --max_seq_length=256 --train_batch_size=1 --learning_rate=5e-6 --batch=1 --doc_stride=128 --num_train_epochs=3.0 --do_lower_case=False --train_file="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/BioASQ-training8b/training8b.json" --predict_file="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/BioASQ-training8b/phaseA_8b_01.json" --output_dir="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/BioASQ-training8b"


python run_list.py \
    --do_train=True \
    --do_predict=True \
    --vocab_file="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/biobert_v1.1_pubmed/vocab.txt" \
    --bert_config_file="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/biobert_v1.1_pubmed/bert_config.json" \
    --init_checkpoint="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/biobert_v1.1_pubmed/model.ckpt-1000000" \
    --max_seq_length=500 \
    --train_batch_size=1 \
    --learning_rate=5e-6 \
    --batch=1 \
    --doc_stride=128 \
    --num_train_epochs=3.0 \
    --do_lower_case=False \
    --train_file="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/BioASQ-training8b/training8b.json" \
    --predict_file="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/BioASQ-training8b/phaseA_8b_01.json" \
    --output_dir="C:/Users/Elias/Documents/TU_Wien/M4.Semester/AIR/BioASQ-training8b" \
```

### Evaluation
We already have the built-in evaluation code. However, this part suggests only for the evaluation part.
The predictions will be saved into a file called `predictions.json` and `nbest_predictions.json` in the `output_dir`.
Run transform file (e.g., `transform_n2b_list.py`) in `./biocodes/` folder to convert `nbest_predictions.json` or `predictions.json` to BioASQ JSON format, which will be used for the official evaluation.
Also, the model will select the final answers by the threshold value, {THRESHOLD}.

```bash
python ./biocodes/transform_n2b_list.py \
    --nbest_path={OUTPUT_DIR}/nbest_predictions.json \
    --threshold={THRESHOLD}
    --output_path={OUTPUT_DIR}
```

This will generate `BioASQform_BioASQ-answer.json` in {OUTPUT_DIR}.
Clone [Evaluation](https://github.com/BioASQ/Evaluation-Measures) code from BioASQ github and run evaluation code on `Evaluation-Measures` directory.
Please note that you should put 5 as parameter for -e if you are evaluating the system for BioASQ 5b/6b/7b/8b dataset.

```bash
cd Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
    $(DATA_DIR)/test/7B1_golden.json \
    $(OUTPUT_DIR)/BioASQform_BioASQ-answer.json
```

This will give you the below scores.
The evaluation scores are sequentially recurs to  YesNo Acc, Factoid Strict Acc, Factoid Lenient Acc, Factoid MRR, **List Prec, List Rec, List F1,** YesNo macroF1, YesNo F1 yes, YesNo F1 no.
```bash
0.6043 0.4881 0.4998
```
