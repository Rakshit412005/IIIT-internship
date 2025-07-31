import matplotlib
matplotlib.use('Agg')  # For headless (no-GUI) environments

import matplotlib.pyplot as plt
import csv
import os

# === Ask for input ===
filename = input("📄 Enter the CSV filename (e.g., vm.csv, cpu.csv, mem.csv): ").strip()

if not os.path.exists(filename):
    print(f"❌ File not found: {filename}")
    exit(1)

# === Metric-specific label map ===
labels = {
    "cpu":   ("CPU Utilization Over Time", "CPU Busy (%)", "cpu_plot.png", 2),
    "mem":   ("Memory Usage Change Over Time", "Memory Change (MB)", "mem_plot.png", 2),
    "disk":  ("Disk Busy Rate Over Time", "Disk Busy (%)", "disk_plot.png", 2),
    "swap":  ("Swap Usage Over Time", "Swap Used (MB)", "swap_plot.png", 2),
    "proc":  ("Process Switches Over Time", "Context Switches/sec", "proc_plot.png", 1),
    "vm":    ("Virtual Memory Paging Over Time", "Pages/sec", "vm_plot.png", 2),
}

key = os.path.basename(filename).split('.')[0].lower()
if key not in labels:
    print("❌ Unknown metric file. Please name it like cpu.csv, mem.csv, etc.")
    exit(1)

title, ylabel, output_name, column_index = labels[key]

# === Read and parse data ===
y = []
with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) <= column_index:
            continue
        try:
            y.append(float(row[column_index]))
        except ValueError:
            continue

if not y:
    print("❌ No numeric data found in the specified column.")
    exit(1)

# === Handle memory specially: plot delta ===
if key == "mem":
    base = y[0]
    y = [val - base for val in y]

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(y, color='blue', label=ylabel)
plt.title(title)
plt.xlabel('Time (snapshots)')
plt.ylabel(ylabel)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_name)
print(f"✅ Graph saved as {output_name}")
