from os.path import dirname

import numpy as np
import pandas as pd
import torch
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.preprocessing import LabelEncoder

#from tensorflow.keras.utils import to_categorical

from LSTMModel import LSTM

window_size = 3

with open("train.log", "r") as file:
    logs = file.readlines()

config = TemplateMinerConfig()
config.load(f"{dirname(__file__)}/drain3.ini")
persistence = FilePersistence("drain3_state.bin")
template_miner = TemplateMiner(persistence, config)

rows = []
for log in logs:
    parts = log.split(" ")
    level = parts[2]
    msg = " ".join(parts[3:])
    rows.append({
        "Level": level,
        "Content": msg
    })

df = pd.DataFrame(rows)

df["EventId"] = None
df["EventTemplate"] = None

for i, row in df.iterrows():
    result = template_miner.add_log_message(row["Content"])
    if result is not None:
        df.at[i, "EventId"] = result["cluster_id"]
        df.at[i, "EventTemplate"] = result["template_mined"]

levels = df["Level"].unique()
level_to_id = {lvl: idx for idx, lvl in enumerate(levels)}
df["LevelId"] = df["Level"].map(level_to_id)

le_event = LabelEncoder()
df['EventId'] = le_event.fit_transform(df['EventId'])

X_event, X_level, y = [], [], []

event_ids = df["EventId"].tolist()
level_ids = df["LevelId"].tolist()

for i in range(len(df) - window_size):
    X_event.append(event_ids[i:i+window_size])
    X_level.append(level_ids[i:i+window_size])
    y.append(event_ids[i+window_size])

X_event = np.array(X_event)
X_level = np.array(X_level)
y = np.array(y)

# One-hot кодирование событий и уровней
num_events = df["EventId"].nunique()
num_levels = df["LevelId"].nunique()

def one_hot_encode(X, num_classes):
    X_safe = np.where(X >= num_classes, num_classes, X)
    return np.array([np.eye(num_classes + 1)[seq] for seq in X_safe])

X_event_oh = one_hot_encode(X_event, num_events)
X_level_oh = one_hot_encode(X_level, num_levels)

# Конкатенируем по последней оси
X_combined = np.concatenate([X_event_oh, X_level_oh], axis=2)

# Переводим в тензоры PyTorch
X_tensor = torch.tensor(X_combined, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

num_epochs = 100
batch_size = 16

dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = LSTM(input_size=X_tensor.shape[2], hidden_size=64, output_size=num_events)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

################# TEST ###############

with open("test.log") as file:
    logs = file.readlines()

rows = []
for log in logs:
    parts = log.split(" ")
    level = parts[2]
    msg = " ".join(parts[3:])
    rows.append({
        "Level": level,
        "Content": msg
    })

df = pd.DataFrame(rows)

df["EventId"] = None
df["EventTemplate"] = None

for i, row in df.iterrows():
    result = template_miner.match(row["Content"])
    df.at[i, "EventId"] = result.cluster_id if result else -1

df['EventId'] = df['EventId'].map(
    lambda x: le_event.transform([x])[0] if x in le_event.classes_ else num_events
)

df["LevelId"] = df["Level"].map(level_to_id)
df["LevelId"] = df["LevelId"].fillna(-1).astype(int)

X_event_test, X_level_test = [], []

event_ids = df["EventId"].tolist()
level_ids = df["LevelId"].tolist()
y_true = []

for i in range(len(df) - window_size):
    X_event_test.append(event_ids[i:i+window_size])
    X_level_test.append(level_ids[i:i+window_size])
    y_true.append(event_ids[i+window_size])

X_event_test = np.array(X_event_test)
X_level_test = np.array(X_level_test)
y_true = np.array(y_true)

X_event_test_oh = one_hot_encode(X_event_test, num_events)
X_level_test_oh = one_hot_encode(X_level_test, num_levels)

# Конкатенируем по последней оси
X_combined = np.concatenate([X_event_test_oh, X_level_test_oh], axis=2)

# Переводим в тензоры PyTorch
X_tensor = torch.tensor(X_combined, dtype=torch.float32)

model(X_tensor).to('cpu')
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor)
    y_pred_probs = torch.softmax(y_pred, dim=1)  # вероятности
    topk_probs, topk_ids = torch.topk(y_pred_probs, k=3, dim=1)

#y_pred_ids = torch.argmax(y_pred, dim=1).numpy()

results_df = pd.DataFrame({
    "y_true": y_true,
    "top1_pred": topk_ids[:,0].numpy(),
    "top2_pred": topk_ids[:,1].numpy(),
    "top3_pred": topk_ids[:,2].numpy()
})

results_df.to_csv("LSTM result.csv", index=False)
print(results_df)

#with torch.no_grad():
#    pred_logits = model(X_tensor)
#    pred_labels = torch.argmax(pred_logits, dim=1)