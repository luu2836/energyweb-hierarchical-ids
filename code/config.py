import os

# Root folder that holds the datasets. Override with the IIOT_DATA_ROOT
# environment variable, otherwise it defaults to the repository's `data/`.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("IIOT_DATA_ROOT", os.path.join(os.path.dirname(_HERE), "data"))

EDGE_ML = os.path.join(DATA_ROOT, "Edge-IIoTset", "ML-EdgeIIoT-dataset.csv")
TON_NET = os.path.join(DATA_ROOT, "ToN-IoT", "train_test_network.csv")
CIC = os.path.join(DATA_ROOT, "CIC-IDS2017", "cic_capped.csv")
UNSW = os.path.join(DATA_ROOT, "UNSW-NB15", "UNSW_NB15_training-set.csv")

# Energy-domain (power-grid) datasets.
# Power System Intrusion Dataset (digital substation, IEC 61850 / GOOSE).
# Source: Kaggle sequincozes/power-system-intrusion-dataset (Apache-2.0).
IEC_TRAIN = os.path.join(DATA_ROOT, "IEC61850", "Train.csv")
IEC_TEST = os.path.join(DATA_ROOT, "IEC61850", "Test.csv")
# Power System Cybersecurity Dataset (generation..control-centre SCADA/PMU).
# Source: Kaggle colabsss/power-system-cybersecurity-dataset (CC0).
PSCS = os.path.join(DATA_ROOT, "PowerSystemCS", "New_Power_System_Cybersecurity.csv")

PROJ = _HERE
RESULT_DIR = os.path.join(os.path.dirname(_HERE), "outputs")
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 42
