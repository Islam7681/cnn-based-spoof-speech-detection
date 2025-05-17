import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from evaluate import compute_eer, plot_roc_curve, plot_confusion_matrix

def show_results(npz_path, set_name=""):
    data = np.load(npz_path, allow_pickle=True)
    label_key = [k for k in data if 'label' in k][0]
    score_key = [k for k in data if 'score' in k][0]
    labels = data[label_key]
    scores = data[score_key]

    print(f"\n=== {set_name} Results ===")
    eer, fpr, tpr, eer_thr = compute_eer(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    preds = (scores > 0.5).astype(int)
    acc = np.mean(preds == labels)
    cm = confusion_matrix(labels, preds)

    print(f"EER: {eer:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)

    plot_roc_curve(fpr, tpr, eer_thr, eer)
    plot_confusion_matrix(cm)
    plt.figure()
    plt.hist(scores, bins=50)
    plt.xlabel('Model output (prob)')
    plt.ylabel('Count')
    plt.title(f'{set_name} predicted scores')
    plt.tight_layout()
    plt.show()

def show_history(history_pkl_path):
    with open(history_pkl_path, 'rb') as f:
        hist = pickle.load(f)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hist['train_loss'], label='Train Loss')
    plt.plot(hist['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss'); plt.xlabel('Epoch')

    plt.subplot(1,2,2)
    plt.plot(hist['train_acc'], label='Train Acc')
    plt.plot(hist['val_acc'], label='Val Acc')
    plt.legend(); plt.title('Accuracy'); plt.xlabel('Epoch')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base = os.path.dirname(__file__)
    dev_npz = os.path.join(base, "final_eval.npz")
    eval_npz = os.path.join(base, "eval_metrics.npz")
    history_pkl = os.path.join(base, "history.pkl")

    if os.path.isfile(dev_npz):
        show_results(dev_npz, "DEV SET")
    else:
        print("No final_eval.npz found.")

    if os.path.isfile(eval_npz):
        show_results(eval_npz, "EVAL SET")
    else:
        print("No eval_metrics.npz found.")

    if os.path.isfile(history_pkl):
        print("\n=== Learning Curves (Train/Dev) ===")
        show_history(history_pkl)
    else:
        print("No history.pkl found.")