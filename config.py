import os

SEED = 42

# paths for the dataset files and folders
BASE_PATH = os.path.join("data", "LA")
TRAIN_FLAC = os.path.join(BASE_PATH, 'ASVspoof2019_LA_train', 'flac')
DEV_FLAC = os.path.join(BASE_PATH, 'ASVspoof2019_LA_dev', 'flac')
TRAIN_PROTOCOL = os.path.join(BASE_PATH, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt')
DEV_PROTOCOL = os.path.join(BASE_PATH, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt')
EVAL_FLAC = os.path.join(BASE_PATH, 'ASVspoof2019_LA_eval', 'flac')
EVAL_PROTOCOL = os.path.join(BASE_PATH, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.eval.trl.txt')

# some basic audio/feature settings
SR = 16000
MAX_SEC = 4
MAX_LEN = SR * MAX_SEC
N_MELS = 64
N_MFCC = 40
FEATURE_TYPE = 'logmel'  # choose between 'mfcc' or 'logmel'

# training hyperparams
BATCH_SIZE = 32
EPOCHS = 9
LR = 1e-3

DEVICE = 'cpu'  # can also be 'cuda' if you want gpu, but my device does not have gpu, so i went with cpu