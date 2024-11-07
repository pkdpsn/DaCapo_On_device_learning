import subprocess
import psutil
import matplotlib.pyplot as plt
import time
import platform
import os
import resource  # Import the resource module to set memory limits

# Print the OS information
print(f"Operating System: {platform.system()}")
print(f"Version: {platform.version()}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")

# Check if the operating system is Linux
if platform.system() != "Linux":
    print("This script is designed to run only on Linux. Exiting...")
    exit()

# Limit memory usage to 600 MB for the subprocess
def set_memory_limit():
    soft, hard = 3000 * 1024 * 1024, 3050 * 1024 * 1024  # 600 MB limit
    print("Setting memory limit...",soft, hard)
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

# Path to your training script
script_path = 'Tiny_Vgg.py'
print(f"Running script: {script_path}")

# Lists to collect resource usage data
cpu_usage = []
ram_usage_mb = []  # RAM usage in MB
timestamps = []
energy_usage = []  # Energy in Joules

# Function to get energy usage on Linux via powercap interface
def get_energy_usage():
    # energy_file = '/sys/class/powercap/intel-rapl:0/energy_uj'
    # if os.path.exists(energy_file):
    #     with open(energy_file, 'r') as f:
    #         return int(f.read().strip()) / 1e6  # Convert from microjoules to joules
    # return None
    return 1

# Start the subprocess with limited memory
process = subprocess.Popen(
    ['python', script_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    # preexec_fn=set_memory_limit,  # Set memory limit before starting the process
    universal_newlines=True,  # Ensures output is returned as a string
    bufsize=1  # Line-buffered output
)

# Get the PID of the subprocess
pid = process.pid
print("Monitoring process with PID:", pid)
proc = psutil.Process(pid)
# proc.cpu_affinity([0])

# Monitor the subprocess
start_time = time.time()
initial_energy = get_energy_usage()  # Initial energy reading
print("Initial energy:", initial_energy)
try:
    # Create iterators to read stdout and stderr
    print("Starting monitoring...")
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line.strip())  # Print output from Tiny_Vgg.py
        
        try:
            proc = psutil.Process(pid)

            # Collect CPU and RAM usage data
            cpu_usage.append(proc.cpu_percent(interval=1))  # CPU percent since last call
            ram_usage_mb.append(max(proc.memory_info().rss / (1024 * 1024)-431,0))  # RAM usage in MB
            timestamps.append(time.time() - start_time)  # Elapsed time

            # Collect energy usage data if available
            current_energy = get_energy_usage()
            if current_energy is not None and initial_energy is not None:
                energy_usage.append(current_energy - initial_energy)

        except psutil.NoSuchProcess:
            print("The process has terminated unexpectedly.")
            break  # Exit the loop if the process is no longer available

        time.sleep(1)  # Wait before next measurement

    process.stdout.close()

except KeyboardInterrupt:
    print("Monitoring interrupted.")

except Exception as e:
    print("An error occurred during monitoring:", e)
# Ensure the process has completed
# process.wait()

# Plot the collected data
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Plot CPU Usage
ax1.plot(timestamps, cpu_usage, label='CPU Usage (%)', color='blue')
ax1.set_title('CPU Usage')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('CPU Usage (%)')

# Plot RAM Usage
ax2.plot(timestamps, ram_usage_mb, label='RAM Usage (MB)', color='orange')
ax2.set_title('RAM Usage')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('RAM Usage (MB)')

# Plot Energy Consumption if available
if energy_usage:
    ax3.plot(timestamps, energy_usage, label='Energy Usage (J)', color='green')
    ax3.set_title('Energy Consumption')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (Joules)')
else:
    ax3.text(0.5, 0.5, 'Energy data unavailable', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    ax3.set_title('Energy Consumption')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (Joules)')

fig.tight_layout()
# plt.show()
plt.savefig('output.png')
