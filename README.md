# AGNews LoRA Finetuning

This repository contains a lightweight fine-tuning approach using LoRA (Low-Rank Adaptation) for text classification on the AGNews dataset.

## Structure
- `notebook/`: Final training and evaluation notebook
- `output/submission.csv`: File ready for Kaggle submission
- `output/logs/`: Training logs (currently empty)
- `output/graphs/`: Visuals like loss curves

## How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/agnews-lora-classifier.git
   cd agnews-lora-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook notebook/AGNews_LoRA_TrainValSplit_Eval.ipynb
   ```

4. Find the file at `output/submission.csv`
