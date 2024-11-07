import psutil
import time
def monitor_processes(pids):
    while True:
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                # Use a small delay to let CPU stats accumulate
                cpu_usage = proc.cpu_percent(interval=1.0)  # Measure CPU over 1 second
                memory_usage = proc.memory_info().rss / 1024 ** 2  # Memory usage in MB
                
                print(f"Process {pid}: CPU usage: {cpu_usage}%, Memory usage: {memory_usage:.2f} MB")
            except psutil.NoSuchProcess:
                print(f"Process {pid} has terminated.")
        time.sleep(2)  # Monitor every 2 seconds
monitor_processes([15824])