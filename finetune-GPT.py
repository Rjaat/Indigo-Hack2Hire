

# import pandas as pd
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
# from datasets import Dataset

# # Load dataset
# df = pd.read_csv('dataset.csv')

# # Load pre-trained model and tokenizer
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# # Set pad token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = '[PAD]'

# # Resize model embeddings to account for new tokens
# model.resize_token_embeddings(len(tokenizer))

# # Prepare dataset
# def preprocess_function(examples):
#     return tokenizer(examples['context'], truncation=True, padding='max_length', max_length=512)

# # Convert DataFrame to Dataset
# dataset = Dataset.from_pandas(df)
# dataset = dataset.map(preprocess_function, batched=True)

# # Check token indices (optional)
# sample_text = df['context'].iloc[0]
# encoded_input = tokenizer(sample_text, return_tensors='pt')
# print(f"Sample encoded input: {encoded_input}")

# # Create data collator
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./gpt2_qa_model",
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset,
# )

# # Train the model
# trainer.train()

# # Save the model
# trainer.save_model()




import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
df = pd.read_csv('dataset.csv')

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'

# Resize model embeddings to account for new tokens
model.resize_token_embeddings(len(tokenizer))

# Prepare dataset
def preprocess_function(examples):
    return tokenizer(examples['context'], truncation=True, padding='max_length', max_length=512)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(preprocess_function, batched=True)

# Check token indices (optional)
sample_text = df['context'].iloc[0]
encoded_input = tokenizer(sample_text, return_tensors='pt')
print(f"Sample encoded input: {encoded_input}")

# Create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments with TensorBoard logging enabled
training_args = TrainingArguments(
    output_dir="./gpt2_qa_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./GPT-logs',              # Directory for TensorBoard logs
    logging_steps=500,                 # Log every 500 steps
    report_to='tensorboard'            # Report metrics to TensorBoard
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model()
