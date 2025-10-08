import re
from multiprocessing import freeze_support
from os.path import dirname
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from loglizer import dataloader
from loglizer.preprocessing import Vectorizer, Iterator

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

from loglizer.models import DeepLog

batch_size = 32
hidden_size = 32
num_directions = 2
topk = 5
train_ratio = 0.2
window_size = 3
epoches = 2
num_workers = 2
device = 0
log_file="structured_train_log.csv"
label_file="label_train_log.csv"

if __name__ == "__main__":
    freeze_support()

# --- Drain3 setup ---
    config = TemplateMinerConfig()
    config.load(f"{dirname(__file__)}/drain3.ini")
    persistence = FilePersistence("drain3_state.bin")
    template_miner = TemplateMiner(persistence, config)

# --- Prepare train logs ---
    with open("train.log", "r") as file:
        logs = file.readlines()

    rows = []
    for index, line in enumerate(logs):
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", line)
        date = date_match.group(0) if date_match else None
    #date = "".join(date.split("-"))

        time_match = re.search(r"\d{2}:\d{2}:\d{2}", line)
        time = time_match.group(0) if time_match else None
    #time = time.replace(":", "")

    # Уровень лога и компонент
        parts = line.split()
        level = parts[2] if len(parts) > 2 else None
        component = parts[3].rstrip(":") if len(parts) > 3 else None

    # Сообщение — всё, что после компонента
        msg = " ".join(parts[4:]) if len(parts) > 4 else None
        msg += f" blk_-{"".join(date.split("-"))}{"".join(time.split(":")[0:2])}00 "

        rows.append({
            "LineId": index,
            "Date": date,
            "Time": time,
            "Level": level,
            "Component": component,
            "Content": msg
        })

    df = pd.DataFrame(rows)

    event_map = {}
    event_counter = 0

    df["EventId"] = None
    df["EventTemplate"] = None

    for i, row in df.iterrows():
        result = template_miner.add_log_message(row["Content"])
        template = result["template_mined"]
        if template not in event_map:
            event_counter += 1
            event_map[template] = f"E{event_counter}"
        df.at[i, "EventId"] = event_map[template]
        df.at[i, "EventTemplate"] = template

    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values('Timestamp')

# Определяем часовые сессии
    df['SessionId'] = df['Timestamp'].dt.floor('min').dt.strftime('%Y%m%d%H%M%S').astype(str)  # округление до часа

# Группируем по SessionId и формируем EventSequence
    sessions = (
        df.groupby('SessionId')['EventId']
        .apply(list)
        .reset_index(name='EventSequence')
    )

# Формируем Label файл (по умолчанию все 0, можно потом пометить известные аномалии)
    labels = pd.DataFrame({
        'BlockId': 'blk_-' + sessions['SessionId'],
        'Label': 'Normal'
    })

    df.to_csv(log_file, index=False)

    labels.to_csv(label_file, index=False)

    (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = dataloader.load_HDFS(log_file,label_file=label_file, window='session', window_size=window_size, train_ratio=train_ratio, split_type='uniform')

    feature_extractor = Vectorizer()

    train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
    test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

    train_loader = Iterator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers).iter
    test_loader = Iterator(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers).iter

    model = DeepLog(num_labels=feature_extractor.num_labels, hidden_size=hidden_size, num_directions=num_directions,
                topk=topk, device=device)
    model.fit(train_loader, epoches)

    #metrics = model.evaluate(train_loader)
    #scores = model.evaluate(test_loader)
    #threshold = np.array(scores).mean() + 3 * np.array(scores).std()

############################################################

    with open("test.log", "r") as file:
        logs = file.readlines()

    rows = []
    for index, line in enumerate(logs):
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", line)
        date = date_match.group(0) if date_match else None
        # date = "".join(date.split("-"))

        time_match = re.search(r"\d{2}:\d{2}:\d{2}", line)
        time = time_match.group(0) if time_match else None
        # time = time.replace(":", "")

        # Уровень лога и компонент
        parts = line.split()
        level = parts[2] if len(parts) > 2 else None
        component = parts[3].rstrip(":") if len(parts) > 3 else None

        # Сообщение — всё, что после компонента
        msg = " ".join(parts[4:]) if len(parts) > 4 else None
        msg += f" blk_-{"".join(date.split("-"))}{"".join(time.split(":")[0:2])}00 "

        rows.append({
            "LineId": index,
            "Date": date,
            "Time": time,
            "Level": level,
            "Component": component,
            "Content": msg
        })

    df = pd.DataFrame(rows)

    event_map = {}
    event_counter = 0

    df["EventId"] = None
    df["EventTemplate"] = None

    for i, row in df.iterrows():
        result = template_miner.match(row["Content"])
        if result is not None:
            df.at[i, "EventId"] = result.cluster_id
            df.at[i, "EventTemplate"] = result.get_template()
        else:
            df.at[i, "EventId"] = "Unknown"
            df.at[i, "EventTemplate"] = "NoMatch"

    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values('Timestamp')
    df['SessionId'] = df['Timestamp'].dt.floor('min').dt.strftime('%Y%m%d%H%M%S').astype(str)

    test_log_file="structured_test_log.csv"
    df.to_csv(test_log_file, index=False)

    # Группируем по SessionId и формируем окна
    X_windows = []
    session_ids = []

    for session_id, group in df.groupby('SessionId'):
        events = group['EventId'].tolist()
        for i in range(len(events) - window_size):
            X_windows.append(events[i:i + window_size])
            session_ids.append(session_id)  # повторяем SessionId для каждого окна

    # Формируем DataFrame
    df_windows = pd.DataFrame({
        "SessionId": session_ids,  # теперь колонка есть
        "EventSequence": X_windows,
        "window_y": [0] * len(X_windows),  # все нули, если нет меток
        "y": [0] * len(X_windows)  # все нули
    })

    # Преобразуем с помощью feature_extractor
    X_transformed = feature_extractor.transform(
        df_windows,
        df_windows['window_y'],
        df_windows['y']
    )

    # 3️⃣ Создаём DataLoader/Iterator
    test_loader = Iterator(X_transformed, batch_size=32, shuffle=False, num_workers=0).iter

    # 4️⃣ Предсказываем
    y_pred = model.predict(test_loader)

    print("Аномалии:", y_pred)