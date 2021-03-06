#@title Default title text
import os
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
import time
import datetime
from tqdm import tqdm
import argparse
import json
import re 
import logging
from preprocessing.data import OKVQA
from models.HypoSelector import HypoSelector
from models.SpanSelector import SpanSelector
logging.basicConfig(level=logging.ERROR)
 
torch.autograd.set_detect_anomaly(True)

MODELS = {
    'span_selector' : SpanSelector,
    'hypo_selector' : HypoSelector,
}

parser = argparse.ArgumentParser()
parser.add_argument('-checkpoint', type=str, default='hypo_selector/', help='Where to store the model checkpoint')
# parser.add_argument('-logs', type=str, default='hypo_selector/', help='Where to store the logs')
# parser.add_argument('-dataset', type=str, default="dataset_hypo_selector", help='Dataset identifier')
parser.add_argument('-model', type=str, default='hypo_selector', choices=MODELS.keys(), help='Model to train')
parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('-batchsize', type=int, default=16, help='Batchsize')
parser.add_argument('-save_after', type=int, default=5, help='Save model after every n-th epoch')

opt = parser.parse_args()
    
dataset_file = "dummy_train.json"
path = "dummy_train"
train_dataset = OKVQA(path,dataset_file)

val_dataset_file = "dummy_val.json"
val_path = "dummy_val"
val_dataset = OKVQA(val_path, val_dataset_file)

print("Dataset Size : ", len(train_dataset), len(val_dataset))
 

batch_size = opt.batchsize

train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )
validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = batch_size 
        )
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
 
def train(model, train_dataloader, optimizer):
    total_train_loss = 0
    total_train_accuracy = 0
    total_train_accuracy_hypo = 0
    total_train_accuracy_span = 0
    model.train()
    
    for batch in train_dataloader:
 
        optimizer.zero_grad()

        loss, (accuracy,accuracy_hypo,accuracy_span), _ = model(batch)
        # print(loss.item())

        loss.backward()

        total_train_loss += loss.item()
        total_train_accuracy += accuracy.item()
        total_train_accuracy_hypo += accuracy_hypo.item()
        total_train_accuracy_span += accuracy_span.item()

        optimizer.step()
    return total_train_loss/len(train_dataloader), total_train_accuracy/len(train_dataset), total_train_accuracy_hypo/len(train_dataset), total_train_accuracy_span/len(train_dataset)
 
def evaluate(model, validation_dataloader):
    total_eval_loss = 0
    total_eval_accuracy = 0
    total_eval_accuracy_hypo = 0
    total_eval_accuracy_span = 0
    model.eval()

    for batch in validation_dataloader:
        with torch.no_grad():
            loss, (accuracy,accuracy_hypo,accuracy_span), _ = model(batch)
            total_eval_loss += loss.item()
            total_eval_accuracy += accuracy.item()
            total_eval_accuracy_hypo += accuracy_hypo.item()
            total_eval_accuracy_span += accuracy_span.item()

    return total_eval_loss / len(validation_dataloader), total_eval_accuracy / len(val_dataset), total_eval_accuracy_hypo / len(val_dataset), total_eval_accuracy_span / len(val_dataset)
 
num_labels = 2
hidden_size = 768

model = MODELS[opt.model](num_labels, hidden_size)

for param in model.bert.parameters():
    param.requires_grad = False

try:
    for param in model.embedding.parameters():
        param.requires_grad = False
except:
    pass
 
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Trainable Parameters : ", total_params)
# print(model)
device = torch.device('cuda')
print("Using the device : ", device)
model.to(device)
 
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

epochs = opt.epochs
save_after = opt.save_after
model_save_filepath = os.path.join('checkpoints', opt.checkpoint+'/')
logs_save_filepath = os.path.join('logs', opt.checkpoint+'/')
try:
    os.makedirs(model_save_filepath)
    os.makedirs(logs_save_filepath)
except:
    pass
epoch_i = 1
 
tl = []
ta = []
vl = []
va = []
 
while epoch_i <= epochs:
    print('\n\n======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
    t0 = time.time()

    train_loss, train_accuracy, train_accuracy_hypo, train_accuracy_span = train(model, train_dataloader, optimizer)
    valid_loss, valid_accuracy, valid_accuracy_hypo, valid_accuracy_span = evaluate(model, validation_dataloader)

    epoch_time = format_time(time.time() - t0)
    print(f'\nTrain Loss: {train_loss:.3f} | Train accuracy: {train_accuracy*100:.2f}%, {train_accuracy_hypo*100:.2f}%, {train_accuracy_span*100:.2f}%')
    print(f'Validation Loss: {valid_loss:.3f} | Validation accuracy: {valid_accuracy*100:.2f}%, {valid_accuracy_hypo*100:.2f}%, {valid_accuracy_span*100:.2f}%')
    print("Time elapsed for epoch: {:}".format(epoch_time))


    tl.append(train_loss)
    ta.append(train_accuracy)
    vl.append(valid_loss)
    va.append(valid_accuracy)

    if epoch_i % save_after == 0:
        print("Saving checkpoint")
        state = {
          'epoch': epoch_i,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
        }

        torch.save(state, model_save_filepath+str(epoch_i)+".pth")

        with open(logs_save_filepath + "logs.json",'w') as f:
            json.dump({"train_loss" : tl, "train_accuracy" : ta, "valid_loss" : vl, "valid_accuracy" : va},f)
    epoch_i += 1
