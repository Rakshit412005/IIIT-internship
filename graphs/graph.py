import matplotlib
matplotlib.use('Agg')  # For headless (no-GUI) environments

import matplotlib.pyplot as plt
import csv
import os

# === Metric-specific label map ===
labels = {
    "cpu":   ("CPU Utilization Over Time", "CPU Busy (%)", "cpu_compare_plot.png", 2),
    "mem":   ("Memory Usage Change Over Time", "Memory Change (MB)", "mem_compare_plot.png", 2),
    "disk":  ("Disk Busy Rate Over Time", "Disk Busy (%)", "disk_compare_plot.png", 2),
    "swap":  ("Swap Usage Over Time", "Swap Used (MB)", "swap_compare_plot.png", 2),
    "proc":  ("Process Switches Over Time", "Context Switches/sec", "proc_compare_plot.png", 1),
    "vm":    ("Virtual Memory Paging Over Time", "Pages/sec", "vm_compare_plot.png", 2),
}

# === Ask for metric ===
metric = input("📊 Enter the metric name (cpu, mem, disk, swap, proc, vm): ").strip().lower()
if metric not in labels:
    print("❌ Unknown metric. Please choose from cpu, mem, disk, swap, proc, vm.")
    exit(1)

title, ylabel, output_name, column_index = labels[metric]

# === Ask for 3 CSV paths ===
phi_path = input("📄 Enter CSV path for Phi-2: ").strip()
gemma_path = input("📄 Enter CSV path for Gemma: ").strip()
qwen_path = input("📄 Enter CSV path for Qwen: ").strip()

# === Function to read Y values from CSV ===
def read_csv_values(filepath, index, is_mem=False):
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    y = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= index:
                continue
            try:
                y.append(float(row[index]))
            except ValueError:
                continue
    if not y:
        print(f"❌ No numeric data in file: {filepath}")
        return None
    if is_mem:
        base = y[0]
        y = [val - base for val in y]
    return y

# === Load data for all 3 models ===
phi_data   = read_csv_values(phi_path, column_index, is_mem=(metric == "mem"))
gemma_data = read_csv_values(gemma_path, column_index, is_mem=(metric == "mem"))
qwen_data  = read_csv_values(qwen_path, column_index, is_mem=(metric == "mem"))

if not (phi_data and gemma_data and qwen_data):
    print("❌ Failed to read all data. Exiting.")
    exit(1)

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(phi_data, color='red', label='Phi-2.Q4_K_M')
plt.plot(gemma_data, color='green', label='gemma-1.1-7b-it.Q4_K_M')
plt.plot(qwen_data, color='blue', label='Qwen_Qwen3-1.7B-Q4_K_M')
plt.title(title)
plt.xlabel('Time (seconds)')
plt.ylabel(ylabel)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_name)
print(f"✅ Comparison graph saved as {output_name}")
