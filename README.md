# AGNews LoRA Classifier - DL Spring 2025 Project 2

## **Overview**  
This project implements a **parameter-efficient fine-tuning strategy** using **Low-Rank Adaptation (LoRA)** on top of a **RoBERTa-base transformer** for **text classification** on the **AGNews dataset**.  
The final model is designed to be **lightweight**, **rubric-compliant**, and optimized for **robust generalization** on the full Kaggle test set.

The model is evaluated based on its ability to classify news articles into one of four categories:
- World
- Sports
- Business
- Sci/Tech

We utilize **LoRA** to fine-tune only a subset of the model parameters, significantly reducing training complexity and memory footprint, while achieving strong performance.

## **Dataset**  
The **AGNews dataset** contains 120,000 training samples and 7,600 test samples, each with:
- `title` (optional)
- `description` or full `text`
- A category label (`0` to `3`)

The test dataset used for inference (`test_unlabelled.pkl`) was provided by the competition organizers and required accurate ID-label mapping for submission.

## **Architecture & Method**  
The final model uses:
- **Backbone:** `roberta-base` from RoBERTa
- **LoRA Adapter Configuration:**
  - Rank `r=8`, Alpha `16`
  - Target Modules: `query`, `value`
  - Dropout: `0.1`
- **Frozen base model**: Only LoRA layers are trained
- **Sequence length:** 128 tokens (data-informed)
- **Scheduler:** Linear with warmup

We trained for **5 epochs** using **FP16 acceleration** on Colab GPUs.

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

This will:
- Load & preprocess AGNews
- Train the model with LoRA
- Plot training loss & accuracy
- Evaluate on validation set
- Generate a valid `submission.csv` file

## ✅ Model Configuration

```json
{
  "model": "roberta-base",
  "lora": {
    "r": 8,
    "alpha": 16,
    "dropout": 0.1,
    "target_modules": ["query", "value"]
  },
  "training": {
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "seed": 42,
  }
}
```

- **Base Model**: RoBERTa-base  
- **LoRA Params**: Rank `r=8`, Alpha `16`, Dropout `0.1`  
- **Trainable Parameters**: 888,580  
- **Scheduler**: Linear with warmup  
- **Device**: Google Colab (T4 GPU)

## ✅ Final Results

| Metric                   | Value        |
|---------------------------|--------------|
| ✅ **Public Kaggle Score** | **0.84675**   |
| ✅ **Private Kaggle Score**| **0.84150**   |
| ✅ Train Accuracy (val)    | 92.25%       |
| Epochs                    | 5  

## **Acknowledgments**  
This project is built on top of HuggingFace Transformers, PEFT (Parameter-Efficient Fine-Tuning) library, and guided by the Spring 2025 Deep Learning course.

If reused, please cite accordingly.
