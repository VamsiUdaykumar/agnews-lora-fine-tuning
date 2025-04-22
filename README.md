# AGNews LoRA Classifier - DL Spring 2025 Project 2

---

## **Overview**  
This project implements a **parameter-efficient fine-tuning strategy** using **Low-Rank Adaptation (LoRA)** on top of a **RoBERTa-base transformer** for **text classification** on the **AGNews dataset**.  
The final model is designed to be **lightweight**, **rubric-compliant**, and optimized for **robust generalization** on the full Kaggle test set.

The model is evaluated based on its ability to classify news articles into one of four categories:
- World
- Sports
- Business
- Sci/Tech

We utilize **LoRA** to fine-tune only a subset of the model parameters, significantly reducing training complexity and memory footprint, while achieving strong performance.

---

## **Dataset**  
The **AGNews dataset** contains 120,000 training samples and 7,600 test samples, each with:
- `title` (optional)
- `description` or full `text`
- A category label (`0` to `3`)

The test dataset used for inference (`test_unlabelled.pkl`) was provided by the competition organizers and required accurate ID-label mapping for submission.

---

## **Architecture & Method**  
The final model uses:
- **Backbone:** `roberta-base` from HuggingFace
- **LoRA Adapter Configuration:**
  - Rank `r=8`, Alpha `16`
  - Target Modules: `query`, `value`
  - Dropout: `0.1`
- **Frozen base model**: Only LoRA layers are trained
- **Sequence length:** 128 tokens (data-informed)
- **Scheduler:** Linear with warmup

We trained for **5 epochs** using **FP16 acceleration** on Colab GPUs with 90%/10% train/val split.

---

## Structure
- `notebook/`: Code base for training and evaluation
- `output/submission.csv`: CSV file
- `output/logs/`: Training logs
- `output/graphs/`: Visuals like loss curves

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook:
   ```bash
   jupyter notebook notebook/DL-Project-2-LORArmstrong.ipynb
   ```

3. Find the file at `output/submission.csv`
