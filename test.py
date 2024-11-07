import subprocess
import psutil
import matplotlib.pyplot as plt
import time
import numpy as np
import platform
import ctypes

print(f"Operating System: {platform.system()}")
print(f"Version: {platform.version()}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")

# Path to your training script
script_path = 'Tiny_Vgg.py'

print(script_path)

# Lists to collect resource usage data
cpu_usage = []
ram_usage_mb = []  # RAM usage in MB
timestamps = []

# Start the subprocess
# process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
#
# Set the CPU affinity to limit to a single core (e.g., core 0)
proc.cpu_affinity([0])
# Monitor the subprocess
start_time = time.time()
try:
    # Create iterators to read stdout and stderr
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line.strip())  # Print output from Tiny_Vgg.py

        try:
            proc = psutil.Process(pid)
            # print(proc, proc.memory_info().rss/(1024*1024))
            cpu_usage.append(proc.cpu_percent(interval=1))  # CPU percent since last call
            ram_usage_mb.append(max(proc.memory_info().rss / (1024 * 1024)-193.95,0))
            timestamps.append(time.time() - start_time)  # Elapsed time

        except psutil.NoSuchProcess:
            print("The process has terminated unexpectedly.")
            break  # Exit the loop if the process is no longer available
        if cpu_usage[-1] > 10:
            time.sleep((cpu_usage[-1] - 20) / 100)  # Adjust sleep time to reduce usage to 20%

        time.sleep(1)  # Wait before next measurement

    process.stdout.close()

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


# import subprocess
# import psutil
# import matplotlib.pyplot as plt
# import time
# import numpy as np
# import platform
# import ctypes
#
# print(f"Operating System: {platform.system()}")
# print(f"Version: {platform.version()}")
# print(f"Platform: {platform.platform()}")
# print(f"Processor: {platform.processor()}")
#
# # Path to your training script
# script_path = 'Tiny_Vgg.py'
# print(script_path)
#
# # Lists to collect resource usage data
# cpu_usage = []
# ram_usage_mb = []  # RAM usage in MB
# timestamps = []
#
# # Start the subprocess
# process = subprocess.Popen(
#     ['python', script_path],
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     universal_newlines=True,  # Ensures output is returned as a string
#     bufsize=1  # Line-buffered output
# )
#
# # Get the PID of the subprocess
# pid = process.pid
# proc = psutil.Process(pid)
#
# # Set the CPU affinity to limit to a single core (e.g., core 0)
# proc.cpu_affinity([0])
#
# # Monitoring and limiting CPU usage
# start_time = time.time()
# try:
#     for stdout_line in iter(process.stdout.readline, ""):
#         print(stdout_line.strip())  # Print output from Tiny_Vgg.py
#
#         # Monitor CPU and RAM usage
#         cpu_usage.append(proc.cpu_percent(interval=0.1))  # Check CPU usage percentage
#         ram_usage_mb.append(proc.memory_info().rss / (1024 * 1024))  # RAM usage in MB
#         timestamps.append(time.time() - start_time)  # Elapsed time
#
#         # Enforce a 20% CPU usage cap
#         if cpu_usage[-1] > 20:
#             time.sleep((cpu_usage[-1] - 20) / 100)  # Adjust sleep time to reduce usage to 20%
#
#     process.stdout.close()
#
# except KeyboardInterrupt:
#     print("Monitoring interrupted.")
#
# # Plot the collected data
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
#
# ax1.plot(timestamps, cpu_usage, label='CPU Usage (%)', color='blue')
# ax1.set_title('CPU Usage')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('CPU Usage (%)')
#
# ax2.plot(timestamps, ram_usage_mb, label='RAM Usage (MB)', color='orange')
# ax2.set_title('RAM Usage')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('RAM Usage (MB)')
#
# fig.tight_layout()
# plt.show()
