# SummarizerPro-With-HuggingFace on SAMSum Dataset

Fine-tune **Facebook's Bart-Large** model for abstractive text summarization using the **SAMSum Dialogue Dataset**. The project implements training, evaluation, and inference pipelines, leveraging state-of-the-art libraries such as **Hugging Face Transformers**, **Datasets**, and **Weights & Biases** for experiment tracking.

## Overview

Abstractive summarization is a technique that generates concise and coherent summaries by rephrasing and condensing content while retaining its meaning. This project uses the **Bart-Large** pre-trained model to summarize conversational datasets.

---

## Features

- **Model**: [Facebook's Bart-Large](https://huggingface.co/facebook/bart-large-cnn)
- **Dataset**: [SAMSum Dialogue Dataset](https://huggingface.co/datasets/samsum)
- **Metrics**: [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) for evaluation
- **Frameworks**:
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers)
  - [Datasets](https://huggingface.co/docs/datasets)
  - [Evaluate](https://huggingface.co/docs/evaluate)
  - [Weights & Biases (W&B)](https://wandb.ai)
- **Hardware Acceleration**: NVIDIA Tesla T4 GPU

---

## Quick Start

### Prerequisites

Ensure you have the following installed:

- Python >= 3.10
- PyTorch >= 2.0.0
- Hugging Face Transformers
- Hugging Face Datasets
- Evaluate
- Weights & Biases (Optional for experiment tracking)

Install dependencies:
```bash
!pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr evaluate wandb
```

---

### Dataset

The [SAMSum Dataset](https://huggingface.co/datasets/samsum) contains over 16,000 dialogues with human-written summaries. It is specifically designed for summarization tasks.

Load the dataset:
```python
from datasets import load_dataset

dataset_samsum = load_dataset("samsum")
print(dataset_samsum)
```

---

### Model

We use **Bart-Large** ([model link](https://huggingface.co/facebook/bart-large-cnn)), a pre-trained transformer model by Facebook, fine-tuned for CNN/DailyMail summarization tasks. The model is extended to handle the SAMSum dataset.

---

### Training

1. **Data Preprocessing**:
   - Convert dialogues and summaries into tokenized input-output pairs.
   - Use the `convert_examples_to_features` function for processing.

   ```python
   dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)
   ```

2. **Model Training**:
   Training is performed with the following configurations:
   - **Batch Size**: 1 (for training and evaluation)
   - **Gradient Accumulation**: 16
   - **Optimizer**: AdamW
   - **Learning Rate Scheduler**: Warmup

   ```python
   from transformers import Trainer, TrainingArguments

   trainer_args = TrainingArguments(
       output_dir='bart-large-samsum',
       num_train_epochs=1,
       warmup_steps=500,
       per_device_train_batch_size=1,
       evaluation_strategy="steps",
       eval_steps=500,
       gradient_accumulation_steps=16
   )

   trainer = Trainer(
       model=model,
       args=trainer_args,
       tokenizer=tokenizer,
       data_collator=seq2seq_data_collator,
       train_dataset=dataset_samsum_pt["train"],
       eval_dataset=dataset_samsum_pt["validation"]
   )

   trainer.train()
   ```

---

### Evaluation

Evaluation is performed using the **ROUGE** metric, which measures the overlap between generated summaries and reference summaries.

```python
from evaluate import load

rouge_metric = load('rouge')

score = calculate_metric_on_test_ds(
    dataset_samsum["test"], rouge_metric, trainer.model, tokenizer, 
    batch_size=2, column_text="dialogue", column_summary="summary"
)

print(score)
```

---

### Inference

Use the fine-tuned model for summarization:

```python
sample_text = dataset_samsum["test"][0]["dialogue"]
pipe = pipeline("summarization", model="bart-large-samsum-model", tokenizer=tokenizer)

print(pipe(sample_text, **{"num_beams": 4, "max_length": 128})[0]["summary_text"])
```

---

### Results

| Metric       | Score    |
|--------------|----------|
| **ROUGE-1**  | 0.012666 |
| **ROUGE-2**  | 0.000257 |
| **ROUGE-L**  | 0.012640 |
| **ROUGE-Lsum** | 0.012666 |

---

### Model and Tokenizer Saving

Save the trained model and tokenizer for reuse:
```python
model.save_pretrained("bart-large-samsum-model")
tokenizer.save_pretrained("tokenizer")
```

Load the saved model and tokenizer:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("bart-large-samsum-model")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
```

---

## Experiment Tracking

This project uses [Weights & Biases (W&B)](https://wandb.ai) for logging and visualization.

Initialize W&B:
```python
import wandb
wandb.init(project="Summarization-SAMSum")
```

Track training:
- View project dashboard: [W&B Dashboard](https://wandb.ai)

---

## Links and References

- **Model**: [Bart-Large](https://huggingface.co/facebook/bart-large-cnn)
- **Dataset**: [SAMSum](https://huggingface.co/datasets/samsum)
- **Hugging Face Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **Weights & Biases**: [W&B](https://wandb.ai)
- **Evaluate Library**: [Evaluate](https://huggingface.co/docs/evaluate)

---

## Acknowledgments

Special thanks to the Hugging Face team and the creators of the SAMSum dataset for their contributions to open-source AI.

---

This repository demonstrates a deep understanding of summarization tasks, from dataset preparation to fine-tuning and deployment. Feel free to fork and experiment with this codebase. Feedback and contributions are welcome!
