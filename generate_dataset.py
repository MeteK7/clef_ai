import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ── Seed for reproducibility ─────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Output folder ─────────────────────────────────────────────────────────────
OUTPUT_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── User profile ──────────────────────────────────────────────────────────────
USER_ID    = "user_001"
OCCUPATION = "Software Engineer"

# ── Event types ───────────────────────────────────────────────────────────────
event_types = [
    {"Id": 1, "Name": "Work",           "Color": "#1f77b4", "UserId": USER_ID},
    {"Id": 2, "Name": "Practice",       "Color": "#ff7f0e", "UserId": USER_ID},
    {"Id": 3, "Name": "Music Research", "Color": "#2ca02c", "UserId": USER_ID},
    {"Id": 4, "Name": "Personal",       "Color": "#d62728", "UserId": USER_ID},
]
event_types_df = pd.DataFrame(event_types)

# ── Boards ────────────────────────────────────────────────────────────────────
boards = [
    {"Id": 1, "Title": "Work Projects",  "CreatedBy": USER_ID, "DateCreated": datetime.now()},
    {"Id": 2, "Title": "Music Projects", "CreatedBy": USER_ID, "DateCreated": datetime.now()},
]
boards_df = pd.DataFrame(boards)

# ── Board columns ─────────────────────────────────────────────────────────────
columns = [
    {"Id": 1, "Title": "To Do"},
    {"Id": 2, "Title": "In Progress"},
    {"Id": 3, "Title": "Done"},
]
board_columns_df = pd.DataFrame(columns)

# ── BoardColumnMappings (use a plain counter for safe int IDs) ────────────────
mappings     = []
mapping_id   = 1
for board in boards:
    for column in columns:
        mappings.append({
            "Id":           mapping_id,
            "BoardId":      board["Id"],
            "BoardColumnId": column["Id"],
            "DateCreated":  datetime.now(),
            "CreatedBy":    USER_ID,
        })
        mapping_id += 1
board_column_mappings_df = pd.DataFrame(mappings)

# ── Board items ───────────────────────────────────────────────────────────────
board_items = []
for i in range(20):  # increased from 10
    board_items.append({
        "Id":            i + 1,
        "Title":         f"Project Task {i + 1}",
        "Description":   f"Task related to {OCCUPATION} or music integration {i + 1}",
        "StatusId":      random.choice([1, 2, 3]),
        "PriorityId":    random.choice([1, 2, 3]),
        "AssigneeId":    USER_ID,
        "DueDate":       datetime.now() + timedelta(days=random.randint(1, 30)),
        "EstimatedTime": round(random.uniform(1, 8), 2),
        "TimeSpent":     round(random.uniform(0, 4), 2),
        "BoardColumnId": random.choice([1, 2, 3]),
        "BoardId":       random.choice([1, 2]),
        "DateCreated":   datetime.now(),
        "CreatedBy":     USER_ID,
    })
board_items_df = pd.DataFrame(board_items)

# ── Calendar events (increased to 200 for better ML training) ─────────────────
NUM_EVENTS      = 200
board_item_ids  = board_items_df["Id"].tolist()
calendar_events = []

for i in range(NUM_EVENTS):
    start_date     = datetime.now() + timedelta(
        days=random.randint(-60, 60),
        hours=random.randint(7, 20),
    )
    duration_hours = random.choice([0.5, 1, 1.5, 2, 3])
    end_date       = start_date + timedelta(hours=duration_hours)

    importance     = random.choice(["Low", "Medium", "High"])
    event_type_obj = random.choice(event_types)
    is_recurring   = random.choice([True, False])
    linked_item    = random.choice(board_item_ids)

    # ── Realistic attendance probability ──────────────────────────────────────
    prob = 0.50
    if event_type_obj["Name"] == "Work":
        prob += 0.25
        if start_date.weekday() >= 5:   # weekends
            prob -= 0.20
    elif event_type_obj["Name"] in ("Practice", "Music Research"):
        prob += 0.10
    if importance == "High":
        prob += 0.20
    elif importance == "Low":
        prob -= 0.10
    if is_recurring:
        prob += 0.10
    if start_date.hour < 9 or start_date.hour > 18:   # outside business hours
        prob -= 0.10
    prob = float(np.clip(prob, 0.05, 0.95))
    attended = int(random.random() < prob)

    # LinkedTaskCompletionRate: float 0.0–1.0
    completion_rate = round(random.uniform(0.0, 1.0), 2)

    calendar_events.append({
        "Id":            i + 1,
        "Subject":       f"{event_type_obj['Name']} Event {i + 1}",
        "Location":      "Online" if random.random() < 0.5 else "Office",
        "StartDate":     start_date,
        "EndDate":       end_date,
        "AllDayEvent":   False,
        "Importance":    importance,
        "Comment":       f"Notes for event {i + 1}",
        "UserId":        USER_ID,
        "LinkedBoardItemId": linked_item,
        "DateCreated":   datetime.now(),
        "CreatedBy":     USER_ID,
        "EventTypeId":   event_type_obj["Id"],
        "IsRecurring":   is_recurring,
        "RecurrenceRuleJson": '{"FREQ":"WEEKLY"}' if is_recurring else None,

        # Behavioral AI features
        "RescheduleCount":    random.randint(0, 5),
        "AvgDaysRescheduled": round(random.uniform(0.0, 3.0), 2),
        "EditCount":          random.randint(0, 10),
        "ViewSignalValue":    round(random.uniform(0.0, 5.0), 2),

        # Task-linked AI features
        "HasLinkedTask":             1,
        "LinkedTaskReopenCount":     random.randint(0, 3),
        "LinkedTaskStatusChanges":   random.randint(1, 10),
        "LinkedTaskCompletionRate":  completion_rate,

        "Attended": attended,
    })

calendar_events_df = pd.DataFrame(calendar_events)

# ── Export ────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, name: str):
    path = os.path.join(OUTPUT_FOLDER, name)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")

print("Generating dataset…")
save(boards_df,               "dummy_boards.csv")
save(board_columns_df,        "dummy_board_columns.csv")
save(board_column_mappings_df,"dummy_board_column_mappings.csv")
save(board_items_df,          "dummy_board_items.csv")
save(calendar_events_df,      "dummy_calendar_events.csv")
save(event_types_df,          "dummy_event_types.csv")

attended_count = calendar_events_df["Attended"].sum()
print(
    f"\nDone! {NUM_EVENTS} events generated "
    f"({attended_count} attended / {NUM_EVENTS - attended_count} skipped)."
)