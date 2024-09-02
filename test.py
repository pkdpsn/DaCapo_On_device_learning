import subprocess
import psutil
import matplotlib.pyplot as plt
import time

# Path to your training script
script_path = 'Tiny_Vgg.py'

# Lists to collect resource usage data
cpu_usage = []
ram_usage = []
timestamps = []

# Start the subprocess
process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Get the PID of the subprocess
pid = process.pid
print(f"Subprocess started with PID: {pid}")
# Monitor the subprocess
start_time = time.time()
try:
    while process.poll() is None:  # While process is running
        proc = psutil.Process(pid)
        cpu_usage.append(proc.cpu_percent(interval=1))  # CPU percent since last call
        # ram_usage.append(proc.memory_percent())
        ram_usage.append(proc.memory_info().rss / (1024 * 1024))  
        timestamps.append(time.time() - start_time)  # Elapsed time
        time.sleep(1)  # Wait before next measurement

except KeyboardInterrupt:
    print("Monitoring interrupted.")

# Ensure the process has completed
process.wait()

# Plot the collected data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.plot(timestamps, cpu_usage, label='CPU Usage (%)', color='blue')
ax1.set_title('CPU Usage')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('CPU Usage (%)')

ax2.plot(timestamps, ram_usage, label='RAM Usage (%)', color='orange')
ax2.set_title('RAM Usage')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('RAM Usage (%)')

fig.tight_layout()
plt.show()
