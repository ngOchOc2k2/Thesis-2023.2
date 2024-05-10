# Continual Relation Extraction
Implementation of the research paper **Improving Rehearsal-free Continual Relation Extraction with Within-Task Variance Awareness** ([EMNLP 2023](https://2023.emnlp.org/))
This is also my graduate thesis.

Status: **Pending**

## Paper
Please refer to [report file](report.pdf) for the full report

## Data Flow Diagram
<div align="center">
<img src=images/DataFlowDiagram.png width=80% />
</div>

## Environment
The system I used and tested in
- Google Colab Server
- NVIDIA A100 Tensor Core GPU with 80 GB memory
- PyTorch 2.0.1
- Huggingface Transformer 4.30.1

## Dependencies
Pre-trained BERT weights:
* Download *bert-base-uncased* into the *datasets/* directory [[google drive]](https://drive.google.com/drive/folders/1BGNdXrxy6W_sWaI9DasykTj36sMOoOGK)

## Run the experiments
To run the experiments, please view the [run](run.ipynb) jupyter ipython notebook

**FewRel**
```
!python3 run_continual.py
  --dataname FewRel
  --encoder_epochs 50
  --encoder_lr 2e-5
  --prompt_pool_epochs 20
  --prompt_pool_lr 1e-4
  --classifier_epochs 500
  --classifier_lr 5e-5
  --replay_epochs 100
  --seed 2021
```

**TACRED**
```
!python3 run_continual.py
  --dataname TACRED
  --encoder_epochs 50
  --encoder_lr 2e-5
  --prompt_pool_epochs 20
  --prompt_pool_lr 2e-5
  --classifier_epochs 500
  --classifier_lr 2e-5
  --replay_epochs 100
  --seed 2021
```

## Experiments results
Please refer to [result folder](results) for the full report

The results were generated using 5 random seeds (2021, 2121, 2221, 2321, 2421) as benchmarks in CRE baselines for a fair comparison
