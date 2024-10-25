import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict, Dataset

# Load the example dataset from the JSON file
with open("example_dataset.json", "r") as f:
    data = json.load(f)

# Convert the JSON data to Hugging Face datasets
train_dataset = Dataset.from_pandas(pd.DataFrame(data["train"]))
validation_dataset = Dataset.from_pandas(pd.DataFrame(data["validation"]))
datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})

# Load the tokenizer and model
model_name = "gpt2"  # Replace with the specific model name if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name)

# Resize model embeddings if new tokens are added
model.resize_token_embeddings(len(tokenizer))

# Preprocess the datasets
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = datasets.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("path_to_save_your_model")  # Replace with your desired path
