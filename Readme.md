# Deepfake Audio Detection with CNNs (ASVspoof2019 LA)

This project trains and evaluates a Convolutional Neural Network (CNN) for detecting spoofed (fake) vs. bonafide (real) speech using the ASVspoof2019 Logical Access (LA) dataset.

---

## Dataset Setup

To run this locally, you’ll need to download the dataset (approx. 8 GB), as it’s not included in the repo.

1. Go to: https://datashare.ed.ac.uk/handle/10283/3336  
2. Download the file named **`LA.zip`**  
3. Unzip it and place the full `ASVspoof2019_LA` folder inside the `data/` folder in this repo (which should currently be empty).  
   Your structure should look like this:

```
data/
└── LA/
    ├── ASVspoof2019_LA_train/
    ├── ASVspoof2019_LA_dev/
    ├── ASVspoof2019_LA_eval/
    └── ASVspoof2019_LA_cm_protocols/
    ... etc
```

---

## How to Train

To train the model, simply run:

```bash
python main.py
```

### What it does:
- Trains on the **train** set, evaluates on the **dev** set after each epoch  
- Saves the trained model to `CNNAudioNet_final.pth`  
- Logs loss/accuracy in `history.pkl`  
- Saves dev scores & confusion matrix in `final_eval.npz`

---

## How to Run Evaluation on Eval (Test) Set

To evaluate on the **unseen evaluation set**, run:

```bash
python run_eval_only.py
```

This will:
- Load your trained model and normalization stats  
- Run prediction on the eval set  
- Save all results to `/eval_metrics.npz`

---

### Viewing Saved Results and Plots

To view all evaluation metrics, learning curves, and confusion matrices, run:

```bash
python report_metrics/show_all_metrics.py
```

**Note:**  
Make sure all result files (e.g., `history.pkl`, `final_eval.npz`, `eval_metrics.npz`, `CNNAudioNet_final.pth`) are placed inside the `report_metrics/` folder.

If you trained or evaluated separately, **move those saved files** into `report_metrics/` before running this script to ensure everything loads correctly.

---


## Configuration

You can change feature types (`logmel` or `mfcc`), training options, or device settings in:

```
config.py
```

---

## Dependencies

All required Python packages are listed in:

```
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

If anything crashes:
- Double-check the **data paths** in `config.py`  
- Make sure the **feature type** matches what you trained with (`logmel` or `mfcc`)  
- Ensure you downloaded and placed the dataset correctly  

---