# import pandas as pd
# from transformers import BertTokenizer, BertForQuestionAnswering
# from torch.utils.data import Dataset, DataLoader
# import torch

# df = pd.read_csv('dataset.csv')

# # Load pre-trained model and tokenizer
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForQuestionAnswering.from_pretrained(model_name)

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
#         encoding = self.tokenizer.encode_plus(
#             question,
#             answer,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_token_type_ids=True,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'token_type_ids': encoding['token_type_ids'].flatten(),
#             'start_positions': torch.tensor([0]),  # Placeholder
#             'end_positions': torch.tensor([0])  # Placeholder
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
#         start_positions = batch['start_positions'].to(device)
#         end_positions = batch['end_positions'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

# # Save the model
# model.save_pretrained("./bert_qa_model")




import pandas as pd
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Load dataset
df = pd.read_csv('dataset.csv')

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

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
        context = self.data.iloc[idx]['context']
        answer = self.data.iloc[idx]['answer']

        # Tokenize the context and question
        encoding = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        # Find start and end positions of the answer
        start_char = context.find(answer)
        end_char = start_char + len(answer)

        # Handle case where the answer is not in the context
        if start_char == -1:
            start_position = end_position = 0
        else:
            offset_mapping = encoding['offset_mapping'].squeeze().tolist()
            start_position = 0
            end_position = 0
            
            for i, (start, end) in enumerate(offset_mapping):
                if start_char >= start and start_char <= end:
                    start_position = i
                if end_char >= start and end_char <= end:
                    end_position = i

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }

# Create dataset and dataloader
train_dataset = QADataset(df, tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Set up TensorBoard writer
writer = SummaryWriter(log_dir='BERT-logs')  # Custom path for logs

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = epoch_loss / len(train_loader)
    writer.add_scalar('Training Loss BERT', avg_loss, epoch + 1)
    print(f"Epoch {epoch + 1} loss: {avg_loss}")

# Save the model and TensorBoard logs
model.save_pretrained("./bert_qa_model")
tokenizer.save_pretrained("./bert_qa_model")
writer.close()
