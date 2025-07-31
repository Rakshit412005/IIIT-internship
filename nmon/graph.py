import matplotlib
matplotlib.use('Agg')  # For headless (no-GUI) environments

import matplotlib.pyplot as plt
import csv

# === File path ===
filename = 'cpu_all.csv'

# === Column index for CPU %busy ===
cpu_busy_column = 2  # Adjust based on your file

y = []
with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) <= cpu_busy_column or not row[cpu_busy_column].replace('.', '', 1).isdigit():
            continue
        y.append(float(row[cpu_busy_column]))

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(y, color='blue', label='%CPU Busy')
plt.title('CPU Utilization Over Time')
plt.xlabel('Time (snapshots)')
plt.ylabel('CPU Busy (%)')
plt.grid(True)
plt.legend()
plt.savefig('cpu_utilization.png')
