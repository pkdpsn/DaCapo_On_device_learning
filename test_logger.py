import subprocess
import psutil
import matplotlib.pyplot as plt
import time
import numpy as np
import platform
import os
import logging

# Set up the logger with a file path in the current project directory
log_directory = "logs"
log_file_path = os.path.join(log_directory, "3C_2F.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger()

# Log system information
logger.info(f"Operating System: {platform.system()}")
logger.info(f"Version: {platform.version()}")
logger.info(f"Platform: {platform.platform()}")
logger.info(f"Processor: {platform.processor()}")

print(f"Operating System: {platform.system()}")
print(f"Version: {platform.version()}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")

# Path to your training script
script_path = 'Tiny_Vgg.py'
logger.info(f"Script Path: {script_path}")

# Lists to collect resource usage data
cpu_usage = []
ram_usage_mb = []  # RAM usage in MB
timestamps = []

# Start the subprocess
process = subprocess.Popen(
    ['python', script_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,  # Ensures output is returned as a string
    bufsize=1  # Line-buffered output
)

# Get the PID of the subprocess
pid = process.pid
proc = psutil.Process(pid)

# Set the CPU affinity to limit to a single core (e.g., core 0)
proc.cpu_affinity([0])
logger.info(f"Started process {pid} with CPU affinity set to core 0")

# Monitor the subprocess
start_time = time.time()
try:
    # Create iterators to read stdout and stderr
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line.strip())  # Print output from Tiny_Vgg.py
        logger.info(f"{stdout_line.strip()}")  # Log script output

        try:
            # Record CPU and RAM usage
            cpu_percent = proc.cpu_percent(interval=1)
            ram_mb = max(proc.memory_info().rss / (1024 * 1024) - 431.5, 0)
            elapsed_time = time.time() - start_time

            cpu_usage.append(cpu_percent)
            ram_usage_mb.append(ram_mb)
            timestamps.append(elapsed_time)

            logger.info(f"RAM Usage: {ram_mb} MB")

        except psutil.NoSuchProcess:
            logger.warning("The process has terminated unexpectedly.")
            break  # Exit the loop if the process is no longer available

        if cpu_usage[-1] > 10:
            time.sleep((cpu_usage[-1] - 20) / 100)  # Adjust sleep time to reduce usage to 20%

        time.sleep(1)  # Wait before next measurement

    process.stdout.close()

except KeyboardInterrupt:
    logger.info("Monitoring interrupted by user.")

# Ensure the process has completed
process.wait()

# Plot the collected data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.plot(timestamps, cpu_usage, label='CPU Usage (%)', color='blue')
ax1.set_title('CPU Usage')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('CPU Usage (%)')

ax2.plot(timestamps, ram_usage_mb, label='RAM Usage (MB)', color='orange')
ax2.set_title('RAM Usage')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('RAM Usage (MB)')

fig.tight_layout()
plt.show()
