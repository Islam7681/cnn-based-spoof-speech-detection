import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# calculate eer/roc stuff from label and score arrays
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idxE = np.nanargmin(abs_diffs)
    eer = (fpr[idxE] + fnr[idxE]) / 2
    return eer, fpr, tpr, thresholds[idxE]

# just draws the roc curve
def plot_roc_curve(fpr, tpr, eer_threshold, eer):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (EER={eer:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
    plt.scatter([eer], [1-eer], c="red", label="EER Point")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (ASVspoof2019 LA)')
    plt.legend()
    plt.grid()
    plt.show()

# training/val loss & accuracy side by side
def plot_learning_curves(history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label="Train")
    plt.plot(history['val_loss'], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label="Train")
    plt.plot(history['val_acc'], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

# show confusion matrix as heatmap
def plot_confusion_matrix(cm):
    import seaborn as sns
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Spoof', 'Bona fide'],
                yticklabels=['Spoof', 'Bona fide'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix")
    plt.show()