import torch
import numpy as np
from tqdm import tqdm
from config import DEVICE

# run through one whole training pass
# will update model weights with backprop
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()  # puts the model in training mode (enables dropout/bn etc)
    running_loss, correct, total = 0., 0, 0
    for feats, labels in tqdm(loader, desc='Train', leave=False):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        # get predicted logits for this batch
        logits = model(feats)
        # compute loss (how far off are we)
        loss = criterion(logits.squeeze(), labels)
        optimizer.zero_grad()   # clear gradients so we don’t accumulate
        loss.backward()         # compute gradients via backprop
        optimizer.step()        # update model weights
        running_loss += loss.item() * feats.size(0)  # add this batch’s loss to total
        preds = (torch.sigmoid(logits.squeeze()) > 0.5).long() # make predictions 0/1
        correct += (preds == labels.long()).sum().item()       # count correct examples
        total += labels.size(0)                                # track the total examples seen
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy   # average values over all batches

# runs one eval pass (so, NO weight updates, just measuring performance)
def eval_one_epoch(model, loader, criterion):
    model.eval()   # turn off dropout/bn moving averages etc
    running_loss, correct, total = 0., 0, 0
    all_labels, all_logits = [], []
    with torch.no_grad():    # no gradients in val
        for feats, labels in tqdm(loader, desc='Valid', leave=False):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits.squeeze(), labels)
            running_loss += loss.item() * feats.size(0)
            preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            # collect all raw outputs and all labels (for roc/auc/eer etc later)
            all_logits.extend(torch.sigmoid(logits.squeeze()).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_logits), np.array(all_labels)