# import pandas as pd
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from torch.utils.data import Dataset, DataLoader
# import torch

# df = pd.read_csv('dataset.csv')

# # Load pre-trained model and tokenizer
# model_name = "t5-small"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# # Prepare the dataset
# class QADataset(Dataset):
#     def __init__(self, dataframe, tokenizer, max_length):
#         self.tokenizer = tokenizer
#         self.data = dataframe
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         question = self.data.iloc[idx]['cleaned_question']
#         answer = self.data.iloc[idx]['cleaned_answer']
#         input_text = f"question: {question} context: {answer}"
#         target_text = answer
        
#         input_encoding = self.tokenizer.encode_plus(
#             input_text,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         target_encoding = self.tokenizer.encode_plus(
#             target_text,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         return {
#             'input_ids': input_encoding['input_ids'].flatten(),
#             'attention_mask': input_encoding['attention_mask'].flatten(),
#             'labels': target_encoding['input_ids'].flatten()
#         }

# # Create dataset and dataloader
# train_dataset = QADataset(df, tokenizer, max_length=512)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # Training loop
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# num_epochs = 3
# for epoch in range(num_epochs):
#     model.train()
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

# # Save the model
# model.save_pretrained("./t5_qa_model")



import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Load dataset
df = pd.read_csv('dataset.csv')

# Load pre-trained model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Prepare the dataset
class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]['question']
        answer = self.data.iloc[idx]['answer']
        input_text = f"question: {question} context: {answer}"
        target_text = answer
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Ignore padding tokens in loss calculation
        labels = target_encoding['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

# Create dataset and dataloader
train_dataset = QADataset(df, tokenizer, max_length=512)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="./T5-logs")

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Log the average loss for this epoch
    writer.add_scalar('Loss/train', avg_loss, epoch + 1)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Save the model
model.save_pretrained("./t5_qa_model")
tokenizer.save_pretrained("./t5_qa_model")

# Save the performance graph using matplotlib
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.savefig('./T5-performance_graph.png')

# Close TensorBoard writer
writer.close()
