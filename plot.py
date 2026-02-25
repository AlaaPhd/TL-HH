import os
import re
import pandas as pd

file_path = "/workspaces/TL-HH/Datasets/TRI_LEVEL_HH_MODELS/InstanceSeparateHH_TriLevel/TriLevel_Q_HH_CMCEE__summary.txt"

with open(file_path, "r") as f:
    lines = f.readlines()

# ===============================
# Extract Final Objectives
# ===============================

final_eff = None
final_div = None
best_LS = None
best_OP = None
best_MA = None
time_to_find = None

ls_data = []
op_data = []
ma_data = []

section = None

for line in lines:
    line = line.strip()

    if "Efficiency:" in line:
        final_eff = int(line.split(":")[1].strip())

    elif "Diversity:" in line:
        final_div = int(line.split(":")[1].strip())

    elif "LS:" in line and "Best Solution" not in line:
        best_LS = int(line.split(":")[1].strip())

    elif "OP:" in line:
        best_OP = int(line.split(":")[1].strip())

    elif "MA:" in line:
        best_MA = int(line.split(":")[1].strip())

    elif "Time to Find:" in line:
        time_to_find = float(line.split(":")[1].replace("seconds", "").strip())

    # Detect sections
    elif "LS Usage Summary" in line:
        section = "LS"

    elif "OP Usage Summary" in line:
        section = "OP"

    elif "MA Usage Summary" in line:
        section = "MA"

    # Parse LS lines
    elif line.startswith("LS") and section == "LS":
        parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        ls_data.append({
            "LS": int(parts[0]),
            "Used": int(parts[1]),
            "Improve": int(parts[2]),
            "Reward": int(parts[3]),
            "Time": float(parts[4])
        })

    # Parse OP lines
    elif line.startswith("OP") and section == "OP":
        parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        op_data.append({
            "OP": int(parts[0]),
            "Used": int(parts[1]),
            "Improve": int(parts[2]),
            "Reward": int(parts[3]),
            "Time": float(parts[4])
        })

    # Parse MA lines
    elif line.startswith("MA") and section == "MA":
        parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        ma_data.append({
            "MA": int(parts[0]),
            "Used": int(parts[1]),
            "Improve": int(parts[2]),
            "Reward": int(parts[3]),
            "Time": float(parts[4])
        })

# Convert to DataFrames
df_ls = pd.DataFrame(ls_data)
df_op = pd.DataFrame(op_data)
df_ma = pd.DataFrame(ma_data)

print("Final Efficiency:", final_eff)
print("Final Diversity:", final_div)
print("Best Config:", best_LS, best_OP, best_MA)
print("Time to Find:", time_to_find)

print("\nLS Data:")
print(df_ls)

print("\nOP Data:")
print(df_op)

print("\nMA Data:")
print(df_ma)
