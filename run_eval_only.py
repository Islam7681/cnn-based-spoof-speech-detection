import torch
from config import (TRAIN_FLAC, TRAIN_PROTOCOL, EVAL_FLAC, EVAL_PROTOCOL, DEVICE)
from dataset import ASVspoofDataset, parse_protocol_with_strings, load_audio, extract_feature
from models import CNNAudioNet
import pickle
import numpy as np
import os

# grab normalization from train set (or load)
train_ds = ASVspoofDataset(TRAIN_FLAC, TRAIN_PROTOCOL, normalize=True)
mean = train_ds.mean
std = train_ds.std

# load your finished model
model = CNNAudioNet().to(DEVICE)
model.load_state_dict(torch.load('report_metrics/CNNAudioNet_final.pth', map_location=DEVICE))

# now just crunch through the eval set and get scores
def eval_metrics_on_eval_set(model, mean, std):
    if not (os.path.exists(EVAL_FLAC) and os.path.exists(EVAL_PROTOCOL)):
        print("EVAL paths do not exist, skipping EVAL metrics computation.")
        return
    utt_ids, label_strs = parse_protocol_with_strings(EVAL_PROTOCOL)
    label_array = np.array([1 if l == 'bonafide' else 0 for l in label_strs])
    print("Unique values and counts in label_array for eval set:",
          np.unique(label_array, return_counts=True))
    preds = []
    model.eval()
    print("\n=== Computing metrics on EVAL SET ===")
    with torch.no_grad():
        for utt_id in utt_ids:
            wav_path = os.path.join(EVAL_FLAC, utt_id + ".flac")
            audio = load_audio(wav_path)
            feat = extract_feature(audio)
            if mean is not None and std is not None:
                feat = (feat - mean[0,:,0][:, None]) / std[0,:,0][:, None]
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            logit = model(feat_tensor)
            score = torch.sigmoid(logit.squeeze()).cpu().item()
            preds.append(score)
    preds_np = np.array(preds)
    # plot + print all metrics
    from evaluate import compute_eer, plot_roc_curve, plot_confusion_matrix
    from sklearn.metrics import roc_auc_score, confusion_matrix
    eer, fpr, tpr, eer_thr = compute_eer(label_array, preds_np)
    roc_auc = roc_auc_score(label_array, preds_np)
    pred_labels = (preds_np > 0.5).astype(int)
    accuracy = np.mean(pred_labels == label_array)
    cm = confusion_matrix(label_array, pred_labels)
    print("\n=== Final Evaluation on EVAL (TEST) Set ===")
    print(f"Eval EER: {eer:.4f}")
    print(f"Eval ROC-AUC: {roc_auc:.4f}")
    print(f"Eval Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", cm)
    plot_roc_curve(fpr, tpr, eer_thr, eer)
    plot_confusion_matrix(cm)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(preds_np, bins=50)
    plt.xlabel('Model output (prob)')
    plt.ylabel('Count')
    plt.title('Eval set predicted scores')
    plt.show()
    np.savez("eval_metrics.npz",
             eval_labels=label_array, eval_scores=preds_np, eval_confusion_matrix=cm)

# just run the thing
eval_metrics_on_eval_set(model, mean, std)