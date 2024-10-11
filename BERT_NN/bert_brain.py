import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from textwrap import wrap

from datasets import load_dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW, get_scheduler
from torch.cuda.amp import GradScaler, autocast

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

am1 = pd.read_csv('AM1_filtered_data.csv')
am1 = am1[['Type', 'Situation']]
am2 = pd.read_csv('AM2.csv') #has all rows
am2 = am2[['Type', 'Situation']]
am3 = pd.read_csv('AM3.csv') #has only headline
am3 = am3.rename(columns={'Headline': 'Situation'})
am3['Type'] = pd.NA
am3 = am3[['Type', 'Situation']]
am4 = pd.read_csv('AM4.csv') #has only headline
am4 = am4.rename(columns={'Headline': 'Situation'})
am4['Type'] = pd.NA
am4 = am4[['Type', 'Situation']]

zaki = pd.read_csv('zaki_filtered_data.csv')
zaki = zaki[['Type', 'Situation']]

sean = pd.read_csv('sean.csv')
sean = sean[['type', 'situations']]
sean = sean.rename(columns={'situations': 'Situation', 'type': 'Type'})

james1 = pd.read_csv('james_3k_filtered_data.csv')
james2 = pd.read_csv('james_6k_filtered_data.csv')
james1 = james1.rename(columns={'situation': 'Situation', 'type': 'Type'})
james2 = james2.rename(columns={'situation': 'Situation', 'type': 'Type'})

all_data = pd.concat([am1, am2, am3, am4, zaki, james1, james2, sean], ignore_index=True)

capitaliq_train, capitaliq_val = train_test_split(all_data, test_size=0.2, random_state=42)

capitaliq_train = capitaliq_train.reset_index(drop=True)
capitaliq_val = capitaliq_val.reset_index(drop=True)

def get_sentiment(text):
    if isinstance(text, str):  # Ensure the input is a string
        score = sia.polarity_scores(text)['compound']  # Get the compound score from VADER
        if score >= 0.05:
            return 1  # Positive sentiment
        elif score <= -0.05:
            return -1  # Negative sentiment
        else:
            return 0  # Neutral sentiment
    else:
        return np.nan  # Return NaN for non-string inputs

capitaliq_train['sentiment'] = capitaliq_train['Situation'].apply(get_sentiment)
capitaliq_val['sentiment'] = capitaliq_val['Situation'].apply(get_sentiment)

capitaliq_train['sentiment'] = capitaliq_train['sentiment'].replace({-1: 0, 0: 1, 1: 2})
capitaliq_val['sentiment'] = capitaliq_val['sentiment'].replace({-1: 0, 0: 1, 1: 2})

external_data_train = pd.read_csv('external_data_train.csv')
external_data_val = pd.read_csv('external_data_test.csv')

external_data_val = external_data_val.rename(columns={'text': 'Situation'})
external_data_train = external_data_train.rename(columns={'text': 'Situation'})

train = pd.concat([external_data_train, capitaliq_train], ignore_index=True)
val = pd.concat([external_data_val, capitaliq_val], ignore_index=True)

types_to_remove = [
    'Fixed Income Offering',
    'Follow-on Equity Offering',
    'Annual General Meeting',
    'Amazon Web Services, Inc.',
    'Amazon.com, Inc. (NasdaqGS:AMZN)',
    'Amazon.com Inc., Investment Arm',
    'Amazon Connect Technology Services (Beijing) Co., Ltd.',
    'Amazon.Com NV Investment Holdings, LLC',
    'Amazon Web Services India Private Limited',
    'Amazon Logistics Inc.',
    'Amazon Web Services, Inc. (NasdaqGS:AMWY)',
    'End of Lock-Up Period',
    'Amazon Web Services, Inc. (NasdaqGS:AMZN)',
    'Amazon.com Services LLC' ,
    'Public Offering Lead Underwriter Change',
    'Shelf Registration Filing',
    'Company Conference Presentation',
    'Conference',
    'Zappos.com LLC',
    'Amazon India Ltd.',
    'Ticker Change',
    'Imdb.Com, Inc.',
    'Board Meeting',
    'Ex-Div Date (Regular)',
    'Index Constituent Add',
    'Index Constituent Drop',
    'Address Change',
    'Whole Foods Market, Inc.',
    '1Life Healthcare, Inc.',
    'Dividend Affirmation',
    'Earnings Release Date',
    'Amazon Web Services India Private Limited',
    'Amazon Digital Services LLC',
    'Deliveroo plc (LSE:ROO)',
    'TWSE:2330',
    'TWSE:6789',
    'NeuralGarage Pvt Ltd',
    'Amazon Robotics LLC',
    'Mgm Interactive Inc.',
    'Zoox Inc.',
    'Alchip Technologies, Limited (TWSE:3661)',
    'Special/Extraordinary Shareholders Meeting',
    'Ex-Div Date (Special)',
    'Dividend Decrease',
    'Delayed SEC Filing'
]

train_final = train[~train['Type'].isin(types_to_remove)]
val_final = val[~val['Type'].isin(types_to_remove)]

train = train.dropna(subset=['Situation'])
val = val.dropna(subset=['Situation'])

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data for both train and validation sets
train_encodings = tokenizer(train['Situation'].tolist(), truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(val['Situation'].tolist(), truncation=True, padding='max_length', max_length=128)

# Convert sentiment labels into tensors
train_labels = torch.tensor(train['sentiment'].tolist(), dtype=torch.long)
val_labels = torch.tensor(val['sentiment'].tolist(), dtype=torch.long)

# Convert the tokenized inputs into PyTorch tensors
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])

val_input_ids = torch.tensor(val_encodings['input_ids'])
val_attention_mask = torch.tensor(val_encodings['attention_mask'])

# Create TensorDatasets for train and validation sets
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

# Create DataLoaders for train and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

scaler = GradScaler()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Move the model to the device
model = model.to(device)

# Function for performing the validation loop
def validate(model, val_dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    # Validation progress bar
    val_progress = tqdm(val_dataloader, desc="Validating", leave=False)

    for batch in val_progress:
        # Unpack the batch (input_ids, attention_mask, labels) and move to device
        input_ids, attention_mask, labels = [t.to(device) for t in batch]  # Move to correct device

        # Cast labels to Long for CrossEntropyLoss
        labels = labels.long()

        # No gradient calculation for validation
        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get predicted class (highest logit value)
        predictions = torch.argmax(logits, dim=-1)

        # Move tensors back to CPU for metric calculation
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')  # Calculate weighted F1 score
    return accuracy, f1

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Set up learning rate scheduler
num_epochs = 8
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps
)

# Training loop with progress bar for each epoch
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Set the model to training mode
    model.train()

    # Create a progress bar for the current epoch
    epoch_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", leave=False)

    for batch in epoch_progress:
        # Unpack the batch
        input_ids, attention_mask, labels = batch

        # Move to device
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device).long()

        # Forward pass with mixed-precision (updated autocast syntax)
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Unscale gradients and perform optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Step learning rate scheduler
        lr_scheduler.step()
        optimizer.zero_grad()

        # Optionally, update the progress bar with the current loss
        epoch_progress.set_postfix({'loss': loss.item()})

    # Run validation after each epoch and compute accuracy and F1 score
    accuracy, f1 = validate(model, val_dataloader, device)
    print(f"Validation Accuracy after epoch {epoch + 1}: {accuracy}")
    print(f"Validation F1 Score after epoch {epoch + 1}: {f1}")

# Final evaluation on the validation set
final_accuracy, final_f1 = validate(model, val_dataloader, device)
print(f"Final Validation Accuracy: {final_accuracy}")
print(f"Final Validation F1 Score: {final_f1}")

torch.save(model.state_dict(), 'NLP_Project_Model92.2.pth')

# !zip -r trained_model.zip trained_model/
# from google.colab import files
# files.download('NLP_Project_Model92.2.zip')