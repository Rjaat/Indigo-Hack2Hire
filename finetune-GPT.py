import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load dataset
df = pd.read_csv('dataset.csv')

# Check for 'context' column
if 'context' not in df.columns:
    raise ValueError("DataFrame must contain a 'context' column.")

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token (GPT-2 does not use pad tokens by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'
model.resize_token_embeddings(len(tokenizer))

# Prepare dataset
def preprocess_function(examples):
    return tokenizer(examples['context'], truncation=True, padding='max_length', max_length=512)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Split dataset into train and validation sets using datasets library
dataset = dataset.train_test_split(test_size=0.1)  # 10% for validation

# Access the split datasets
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Preprocess the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments with TensorBoard logging and evaluation enabled
training_args = TrainingArguments(
    output_dir="./gpt2_qa_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./GPT-logs',              # Directory for TensorBoard logs
    logging_steps=500,                    # Log every 500 steps
    evaluation_strategy="steps",          # Evaluate every `eval_steps` steps
    eval_steps=500,                       # Evaluate every 500 steps
    report_to='tensorboard'               # Report metrics to TensorBoard
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Passing the validation dataset 
)

# Train model
trainer.train()

# Evaluate model
results = trainer.evaluate(eval_dataset=eval_dataset)
print("Evaluation results:", results)

# Save model
trainer.save_model()
