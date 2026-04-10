import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
import os

# --- Create "data" folder if it doesn't exist ---
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# --- USER PROFILE ---
user_id = "user_001"
hobby = "Cello"
occupation = "Software Engineer"

# --- Event types ---
event_types = [
    {"Id": 1, "Name": "Work", "Color": "#1f77b4", "UserId": user_id},
    {"Id": 2, "Name": "Practice", "Color": "#ff7f0e", "UserId": user_id},
    {"Id": 3, "Name": "Music Research", "Color": "#2ca02c", "UserId": user_id},
    {"Id": 4, "Name": "Personal", "Color": "#d62728", "UserId": user_id},
]

event_types_df = pd.DataFrame(event_types)

# --- Boards ---
boards = [
    {"Id": 1, "Title": "Work Projects", "CreatedBy": user_id, "DateCreated": datetime.now()},
    {"Id": 2, "Title": "Music Projects", "CreatedBy": user_id, "DateCreated": datetime.now()},
]

boards_df = pd.DataFrame(boards)

# --- Board Columns ---
columns = [
    {"Id": 1, "Title": "To Do"},
    {"Id": 2, "Title": "In Progress"},
    {"Id": 3, "Title": "Done"},
]

board_columns_df = pd.DataFrame(columns)

# --- BoardColumnMappings ---
mappings = []
for board in boards:
    for column in columns:
        mappings.append({
            "Id": uuid.uuid4().int >> 64,
            "BoardId": board["Id"],
            "BoardColumnId": column["Id"],
            "DateCreated": datetime.now(),
            "CreatedBy": user_id
        })
board_column_mappings_df = pd.DataFrame(mappings)

# --- Board Items ---
board_items = []
item_id_counter = 1
for i in range(10):
    board_items.append({
        "Id": item_id_counter,
        "Title": f"Project Task {i+1}",
        "Description": f"Task related to {occupation} or music integration {i+1}",
        "StatusId": random.choice([1,2,3]),
        "PriorityId": random.choice([1,2,3]),
        "AssigneeId": user_id,
        "DueDate": datetime.now() + timedelta(days=random.randint(1,30)),
        "EstimatedTime": random.uniform(1,8),
        "TimeSpent": random.uniform(0,4),
        "BoardColumnId": random.choice([1,2,3]),
        "BoardId": random.choice([1,2]),
        "DateCreated": datetime.now(),
        "CreatedBy": user_id
    })
    item_id_counter += 1

board_items_df = pd.DataFrame(board_items)

# --- Calendar Events ---
calendar_events = []
event_id_counter = 1
for i in range(50):  # more events for better dataset
    start_date = datetime.now() + timedelta(days=random.randint(-10, 20), hours=random.randint(8,20))
    end_date = start_date + timedelta(hours=random.choice([1,2,3]))
    importance = random.choice(["Low","Medium","High"])
    linked_item = random.choice(board_items_df["Id"].tolist())
    event_type_obj = random.choice(event_types)
    event_type = event_type_obj["Id"]
    is_recurring = random.choice([True, False])

    # --- Realistic attendance label ---
    prob_attend = 0.5
    if event_type_obj["Name"] == "Work":
        prob_attend += 0.3  # more likely to attend work
        if start_date.weekday() >= 5:
            prob_attend -= 0.2  # weekends, less likely
    elif event_type_obj["Name"] in ["Practice", "Music Research"]:
        prob_attend += 0.1
    if importance == "High":
        prob_attend += 0.2
    if is_recurring:
        prob_attend += 0.1

    prob_attend = min(max(prob_attend, 0), 1)
    attended = int(random.random() < prob_attend)

    calendar_events.append({
        "Id": event_id_counter,
        "Subject": f"{event_type_obj['Name']} Event {i+1}",
        "Location": "Online" if random.random() < 0.5 else "Office",
        "StartDate": start_date,
        "EndDate": end_date,
        "AllDayEvent": False,
        "Importance": importance,
        "Comment": f"Notes for event {i+1}",
        "UserId": user_id,
        "LinkedBoardItemId": linked_item,
        "DateCreated": datetime.now(),
        "CreatedBy": user_id,
        "EventTypeId": event_type,
        "IsRecurring": is_recurring,
        "RecurrenceRuleJson": '{"FREQ":"WEEKLY"}' if is_recurring else None,

        # 🔥 NEW AI FEATURES (ADD HERE)
        "RescheduleCount": random.randint(0, 5),
        "AvgDaysRescheduled": random.uniform(0, 3),
        "EditCount": random.randint(0, 10),
        "ViewSignalValue": random.uniform(0, 5),

        "HasLinkedTask": 1,
        "LinkedTaskReopenCount": random.randint(0, 3),
        "LinkedTaskStatusChanges": random.randint(1, 10),
        "LinkedTaskCompletionRate": random.choice([0, 1]),

        "Attended": attended
    })
    event_id_counter += 1

calendar_events_df = pd.DataFrame(calendar_events)

# --- Export to CSV in "data" folder ---
boards_df.to_csv(os.path.join(output_folder, "dummy_boards.csv"), index=False)
board_columns_df.to_csv(os.path.join(output_folder, "dummy_board_columns.csv"), index=False)
board_column_mappings_df.to_csv(os.path.join(output_folder, "dummy_board_column_mappings.csv"), index=False)
board_items_df.to_csv(os.path.join(output_folder, "dummy_board_items.csv"), index=False)
calendar_events_df.to_csv(os.path.join(output_folder, "dummy_calendar_events.csv"), index=False)
event_types_df.to_csv(os.path.join(output_folder, "dummy_event_types.csv"), index=False)

print("Dummy dataset with realistic attendance labels generated successfully in folder 'data'!")