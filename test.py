import subprocess
import psutil
import matplotlib.pyplot as plt
import time
import numpy as np

# Path to your training script
script_path = 'Tiny_Vgg.py'

# Lists to collect resource usage data
cpu_usage = []
ram_usage_mb = []  # RAM usage in MB
timestamps = []

# Start the subprocess
process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Get the PID of the subprocess
pid = process.pid

# Monitor the subprocess
start_time = time.time()
try:
    while process.poll() is None:  # While the process is running
        try:
            proc = psutil.Process(pid)
            print(proc, proc.memory_info().rss/(1024*1024))
            cpu_usage.append(proc.cpu_percent(interval=1))  # CPU percent since last call
            ram_usage_mb.append(proc.memory_info().rss / (1024 * 1024))  # RAM usage in MB
            timestamps.append(time.time() - start_time)  # Elapsed time

        except psutil.NoSuchProcess:
            print("The process has terminated unexpectedly.")
            break  # Exit the loop if the process is no longer available
        
        time.sleep(1)  # Wait before next measurement

except KeyboardInterrupt:
    print("Monitoring interrupted.")

# Ensure the process has completed
# process.wait()

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
