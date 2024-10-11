import pandas
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# List of types to remove
types_to_remove = [
    'Fixed Income Offering', 'Follow-on Equity Offering', 'Annual General Meeting',
    'End of Lock-Up Period', 'Public Offering Lead Underwriter Change',
    'Shelf Registration Filing', 'Company Conference Presentation', 'Conference',
    'Zappos.com LLC', 'Ticker Change', 'Imdb.Com, Inc.', 'Board Meeting',
    'Ex-Div Date (Regular)', 'Index Constituent Add', 'Index Constituent Drop',
    'Address Change', 'Whole Foods Market, Inc.', '1Life Healthcare, Inc.',
    'Dividend Affirmation', 'Earnings Release Date', 'Deliveroo plc (LSE:ROO)',
    'TWSE:2330', 'TWSE:6789', 'NeuralGarage Pvt Ltd', 'Mgm Interactive Inc.',
    'Zoox Inc.', 'Alchip Technologies, Limited (TWSE:3661)',
    'Special/Extraordinary Shareholders Meeting', 'Ex-Div Date (Special)',
    'Dividend Decrease', 'Delayed SEC Filing'
]

# Load the BERT model with the number of labels you used during training (replace with your actual number of labels)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load your trained model parameters and map to CPU if needed
model.load_state_dict(torch.load('NLP_Project_Model92.2.pth', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#filters data and removes
def filter_data(df):
    return df[~df['Type'].isin(types_to_remove)]
def predict(text):
    # Tokenize the input text
    encoded_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'  # Return PyTorch tensors
    )

    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Make predictions
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]  # Get the logits (predicted scores for each class)

    # Get the predicted class (index of the highest score)
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def fixVals(arr):
    for i in range(len(arr)):
        for s in range(len(arr[i])):
            arr[i][s] = arr[i][s] - 1
    return arr

def totSent(arr):
    total = 0
    for s in range(len(arr)):
        total = total + arr[s]
    return total

def fixDate(date):
    month_days = {
        'Jan': 0, 'Feb': 31, 'Mar': 59, 'Apr': 90,
        'May': 120, 'Jun': 151, 'Jul': 181, 'Aug': 212,
        'Sep': 243, 'Oct': 273, 'Nov': 304, 'Dec': 334
    }

    # Split the input date
    x = date.split('-')

    # Get the cumulative day count at the start of the month and add the day
    month = x[0]
    day = int(x[1])
    total_days = month_days.get(month, 0) + day

    return total_days

earnDate = 'Jan-3-2024'
dates = []
#ML Model
rawOutputs = [[1, 1, 2, 2, 2, 2, 0, 2, 0, 1, 2, 2], [1, 0, 2, 1, 2, 2, 1, 0, 2, 1, 2, 0, 2, 2, 1],
              [0, 0, 2, 0, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 1, 1, 1, 0]]
outputs = fixVals(rawOutputs)

def main(df):
    filter_data(df)
    df[:,0] = fixDate([:,0])
    df_dates = df[:, 0]
    raw_data1 = list
    raw_data2 = list
    raw_data3 = list
    raw_data4 = list
    model_outputs1 = np.zeros((1, len(raw_data1)))
    model_outputs2 = np.zeros((1, len(raw_data2)))
    model_outputs3 = np.zeros((1, len(raw_data3)))
    model_outputs4 = np.zeros((1, len(raw_data4)))
    for i in range(len(df[:,0])):
        if (i >= 0 and i < 10):
            raw_data1.append(df[i,2])
        if (i >= 10 and i < 20):
            raw_data1.append(df[i, 2])
        if (i >= 20 and i < 30):
            raw_data1.append(df[i,2])
        if (i >= 30 and i < 40):
            raw_data1.append(df[i,2])
    for i in range(len(raw_data1)):
        model_outputs1[i] = predict(raw_data1[i])
    for i in range(len(raw_data2)):
        model_outputs2[i] = predict(raw_data1[i])
    for i in range(len(raw_data3)):
        model_outputs3[i] = predict(raw_data1[i])
    for i in range(len(raw_data4)):
        model_outputs4[i] = predict(raw_data1[i])
    model_outputs1 = fixVals(model_outputs1)
    model_outputs2 = fixVals(model_outputs2)
    model_outputs3 = fixVals(model_outputs3)
    model_outputs4 = fixVals(model_outputs4)
    data = [totSent(model_outputs1),totSent(model_outputs2),totSent(model_outputs3),totSent(model_outputs4)]
    x_axis = [1,2,3,4]
    plt.plot(data,x_axis)
    plt.show()
        



# #Main
# predicted = predict("The Jubilant Bhartia Group promoters are leading the race to acquire a significant minority stake in The Coca-Cola Company (NYSE:KO)'s India bottling arm, Hindustan Coca-Cola Beverages Private Limited (HCCB). They have signed an exclusivity agreement with Coca-Cola to negotiate the purchase for INR 108,000 million-INR 120,000 million. This move is part of Coca-Cola's strategy to adopt an asset-light model and precedes a planned listing of HCCB. The promoters of the Jubilant Bhartia Group have emerged as frontrunners for a stake in Coca-Cola's India bottling arm, bettering an offer by the Burmans of Dabur India Limited (NSEI:DABUR), as they amp up their bet on the country's evolving consumption patterns and rising disposable income, said people with knowledge of the matter.")
# finDay = fixDate(earnDate)
# startDay = finDay - 3
# time = np.linspace(startDay, finDay, 3)
# sentOvTime = np.zeros(len(outputs))
# for i in range(len(outputs)):
#     sent = totSent(outputs[i])
#     print(sent)
#     sentOvTime[i] = sent
# print(sentOvTime)
# plt.plot(time, sentOvTime)
# plt.show()
