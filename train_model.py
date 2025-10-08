from os.path import dirname
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from deepod.models.dsvdd import DeepSVDD
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# --- Drain3 setup ---
config = TemplateMinerConfig()
config.load(f"{dirname(__file__)}/drain3.ini")
persistence = FilePersistence("drain3_state.bin")
template_miner = TemplateMiner(persistence, config)

# --- Read training logs ---
with open("train.log", "r") as file:
    logs = file.readlines()

event_ids = []
for log in logs:
    res = template_miner.add_log_message(log.strip())
    event_ids.append(res["cluster_id"])

# --- Encode event IDs ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
event_vecs = encoder.fit_transform(np.array(event_ids).reshape(-1, 1))

# --- Build windows ---
window_size = 5
X = [event_vecs[i:i+window_size].flatten() for i in range(len(event_vecs)-window_size)]
X = np.array(X)

# --- Train DeepSVDD ---
model = DeepSVDD(epochs=30, device='cpu')
model.fit(X)

scores_train = model.decision_function(X)
threshold = np.mean(scores_train) + 3 * np.std(scores_train)

# --- Read test logs ---
with open("test.log", "r") as file:
    test_logs = file.readlines()

test_ids = []
for log in test_logs:
    res = template_miner.add_log_message(log.strip())
    test_ids.append(res["cluster_id"])

test_vecs = encoder.transform(np.array(test_ids).reshape(-1, 1))
X_test = [test_vecs[i:i+window_size].flatten() for i in range(len(test_vecs)-window_size)]
X_test = np.array(X_test)

# --- Проверка на аномальных логах ---
scores = model.decision_function(X_test)

anomalies = 0
for index, score in enumerate(scores):
    if score > threshold:
        anomalies += 1
        print(f"Обнаружено аномальное окно, строка {index}")

print(f"Detected anomalies: {anomalies} of {len(X_test)} windows")
