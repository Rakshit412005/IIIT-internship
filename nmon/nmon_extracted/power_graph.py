import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

cpu_file = "cpu.csv"
cpu_col = 2  # Adjust based on your actual CPU % column

cpu_percent = []

with open(cpu_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            cpu_percent.append(float(row[cpu_col]))
        except:
            continue

# Estimate power: idle = 1.5W, full = 6.0W
power = [1.5 + 4.5 * (x / 100) for x in cpu_percent]

plt.figure(figsize=(10, 6))
plt.plot(power, color='red')
plt.title('Estimated Power Consumption Over Time')
plt.xlabel('Time (snapshots)')
plt.ylabel('Power (Watts)')
plt.grid(True)
plt.tight_layout()
plt.savefig("power_estimate.png")
print("✅ Saved estimated power graph as power_estimate.png")

