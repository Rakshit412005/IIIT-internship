import os
import subprocess

# === Ask user for file name ===
nmon_file = input("🔍 Enter your NMON file name (with .nmon extension): ").strip()

# === Check if file exists ===
if not os.path.isfile(nmon_file):
    print(f"❌ File not found: {nmon_file}")
    exit(1)

# === Directory to save extracted data ===
output_dir = "nmon_extracted"
os.makedirs(output_dir, exist_ok=True)

# === NMON section mappings ===
sections = {
    "cpu_all.csv": "CPU_ALL",
    "mem.csv": "MEM",
    "disk.csv": "DISK",
    "swap.csv": "SWAP",
    "proc.csv": "PROC",
    "vm.csv": "VM",
    "time.csv": "ZZZ"
}

# === Extract each section using grep ===
for filename, section in sections.items():
    output_path = os.path.join(output_dir, filename)
    command = f"grep '^{section}' \"{nmon_file}\" > \"{output_path}\""
    result = subprocess.run(command, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print(f"✅ Extracted {section} → {output_path}")
    else:
        print(f"❌ Failed to extract {section}: {result.stderr.decode().strip()}")
