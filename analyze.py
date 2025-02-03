import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import glob
import os

def extract_kernel_padding_model(file_path):
    """Extracts kernel, padding, and model type from the log file header."""
    kernel, padding, model_type = "unknown", "unknown", "unknown"
    with open(file_path, "r") as file:
        for line in file:
            kernel_match = re.search(r"Kernel: (\d+) \| Pad: (\d+)", line)
            model_match = re.search(r"Algo: (\w+)", line)
            if kernel_match:
                kernel, padding = kernel_match.groups()
            if model_match:
                model_type = model_match.group(1)
    return kernel, padding, model_type

def process_log_files(log_files_pattern):
    """Reads multiple log files, aggregates training metrics, and saves results."""
    all_data = []
    output_dir = "aggregated_results"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the results directory exists
    
    # Extract kernel, padding, and model type from the first log file
    sample_file = glob.glob(log_files_pattern)[0]
    kernel, padding, model_type = extract_kernel_padding_model(sample_file)
    
    for file_path in glob.glob(log_files_pattern):
        with open(file_path, "r") as file:
            log_data = file.readlines()
        
        # Define regex pattern for extracting relevant information
        epoch_pattern = re.compile(
            r"Epoch (\d+)/\d+, Loss: ([\d\.]+), Time: ([\d\.]+)s, CPU Mem: ([\d\.]+) MB, "
            r"GPU Mem: ([\d\.]+) MB, CPU Usage: ([\d\.]+)%, GPU Power: ([\d\.]+) W"
        )
        
        # Extract data
        data = []
        for line in log_data:
            match = epoch_pattern.search(line)
            if match:
                epoch, loss, time, cpu_mem, gpu_mem, cpu_usage, gpu_power = match.groups()
                data.append([int(epoch), float(loss), float(time), float(cpu_mem), float(gpu_mem), float(cpu_usage), float(gpu_power)])
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["Epoch", "Loss", "Time (s)", "CPU Memory (MB)", "GPU Memory (MB)", "CPU Usage (%)", "GPU Power (W)"])
        all_data.append(df)
    
    # Combine all runs by averaging values per epoch
    combined_df = pd.concat(all_data).groupby("Epoch").mean()
    
    # Append row with average of all metrics
    avg_row = combined_df.mean().to_frame().T
    avg_row.index = ["Average"]
    combined_df = pd.concat([combined_df, avg_row])
    
    # Generate file names with kernel, padding, and model type info
    csv_filename = os.path.join(output_dir, f"agg_metrics_k{kernel}p{padding}_{model_type}.csv")
    graph_filename = os.path.join(output_dir, f"agg_loss_k{kernel}p{padding}_{model_type}.png")
    
    # Save average metrics to CSV
    combined_df.to_csv(csv_filename)
    
    # Plot and save Loss vs Epoch graph
    plt.figure(figsize=(10, 5))
    plt.plot(combined_df.index[:-1], combined_df.loc[combined_df.index[:-1], "Loss"], marker='o', linestyle='-', label="Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Average Loss vs Epoch (Kernel={kernel}, Padding={padding}, Model={model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(graph_filename)
    plt.show()
    
    print(f"Saved graph: {graph_filename}")
    print(f"Saved average metrics: {csv_filename}")

# Run script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_log.py <log_files_pattern>")
        sys.exit(1)
    log_files_pattern = sys.argv[1]  # Example: "main_6193_*.log"
    process_log_files(log_files_pattern)
