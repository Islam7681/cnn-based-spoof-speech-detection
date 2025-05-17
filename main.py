import torch
import numpy as np
from config import (TRAIN_FLAC, DEV_FLAC, TRAIN_PROTOCOL, DEV_PROTOCOL, 
                    BATCH_SIZE, DEVICE, EPOCHS, LR, SEED)
from dataset import ASVspoofDataset, ASVspoofDevDataset
from torch.utils.data import DataLoader
from models import CNNAudioNet
from train import train_one_epoch, eval_one_epoch
from evaluate import compute_eer, plot_roc_curve, plot_learning_curves, plot_confusion_matrix
from sklearn.metrics import roc_auc_score, confusion_matrix
import random
import pickle
import matplotlib.pyplot as plt
import os

# just seed everything for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# get loaders for train/dev, apply normalization
def get_dataloaders():
    train_ds = ASVspoofDataset(TRAIN_FLAC, TRAIN_PROTOCOL)
    dev_ds = ASVspoofDevDataset(DEV_FLAC, DEV_PROTOCOL, normalize=True)
    dev_ds.set_norm(train_ds.mean, train_ds.std)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, dev_loader, train_ds, dev_ds

# quick sanity report, makes sure shapes/labels/scores are matching
def integrity_report(dev_ds, val_labels, val_scores):
    print("\n=== DATA INTEGRITY REPORT ===")
    print("Len dev_ds:", len(dev_ds))
    print("Len val_labels:", len(val_labels))
    print("True bonafide:", int(np.sum(val_labels)))
    print("True spoof:", len(val_labels) - int(np.sum(val_labels)))
    preds = (val_scores > 0.5).astype(int)
    print("Pred bonafide:", int(preds.sum()))
    print("Pred spoof:", len(preds) - int(preds.sum()))
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels, preds))
    plt.figure()
    plt.hist(val_scores, bins=50)
    plt.xlabel('Model output (prob)')
    plt.ylabel('Count')
    plt.title('Dev set predicted scores')
    plt.show()

def main():
    set_seed(SEED)
    train_loader, dev_loader, train_ds, dev_ds = get_dataloaders()
    model = CNNAudioNet().to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_eer = 1.0

    # training loop – one epoch at a time
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_scores, val_labels = eval_one_epoch(model, dev_loader, criterion)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        eer, fpr, tpr, eer_thr = compute_eer(val_labels, val_scores)
        roc_auc = roc_auc_score(val_labels, val_scores)
        print(f"Epoch {epoch+1:2d}/{EPOCHS}: train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} EER={eer:.4f} ROC-AUC={roc_auc:.4f}")
        if eer < best_eer:
            best_eer = eer
            # uncomment to save best model
            # torch.save(model.state_dict(), "CNNAudioNet_best.pth")
            
    print("\n=== Final Evaluation on Dev Set ===")
    eer, fpr, tpr, eer_thr = compute_eer(val_labels, val_scores)
    roc_auc = roc_auc_score(val_labels, val_scores)
    preds = (val_scores > 0.5).astype(int)
    accuracy = np.mean(preds == val_labels)
    cm = confusion_matrix(val_labels, preds)
    print(f"Dev EER: {eer:.4f}")
    print(f"Dev ROC-AUC: {roc_auc:.4f}")
    print(f"Dev Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", cm)
    plot_learning_curves(history)
    plot_roc_curve(fpr, tpr, eer_thr, eer)
    plot_confusion_matrix(cm)
    integrity_report(dev_ds, val_labels, val_scores)

    # save stuff for later plotting/easy reload
    with open("history.pkl", "wb") as f:
        pickle.dump(history, f)
    np.savez("final_eval.npz", 
             val_labels=val_labels, 
             val_scores=val_scores, 
             confusion_matrix=cm)
    torch.save(model.state_dict(), "CNNAudioNet_final.pth")

if __name__ == '__main__':
    main()