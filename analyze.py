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
            r"Epoch (\d+)/(\d+), Loss: ([\d\.]+), Time: ([\d\.]+)s, CPU Mem: ([\d\.]+) MB, "
            r"GPU Mem: ([\d\.]+) MB, CPU Usage: ([\d\.]+)%, GPU Power: ([\d\.]+) W"
        )
        
        # Extract data
        data = []
        for line in log_data:
            match = epoch_pattern.search(line)
            if match:
                epoch, total_epochs, loss, time, cpu_mem, gpu_mem, cpu_usage, gpu_power = match.groups()
                epoch, total_epochs = int(epoch), int(total_epochs)
                time, gpu_power = float(time), float(gpu_power)
                power_work = (gpu_power * time) / 3600  # Convert seconds to hours for Wh
                data.append([epoch, total_epochs, float(loss), time, float(cpu_mem), float(gpu_mem), float(cpu_usage), gpu_power, power_work])
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["Epoch", "Total Epochs", "Loss", "Time (s)", "CPU Memory (MB)", "GPU Memory (MB)", "CPU Usage (%)", "GPU Power (W)", "Power Work (Wh)"])
        all_data.append(df)
    
    # Combine all runs by averaging values per epoch
    combined_df = pd.concat(all_data).groupby("Epoch").mean()
    combined_df.index = pd.to_numeric(combined_df.index, errors='coerce').dropna().astype(int)  # Ensure epoch index is integer
    
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

    # Select only numeric indices for plotting and xticks
    numeric_indices = combined_df.index[combined_df.index != "Average"]
    plt.plot(numeric_indices, combined_df.loc[numeric_indices, "Loss"], marker=None, linestyle='-', label="Average Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Average Loss vs Epoch (Kernel={kernel}, Padding={padding}, Model={model_type})")

    # Use only numeric indices for xticks and adjust range
    max_epoch = numeric_indices.max()
    plt.xticks(range(0, max_epoch + 1, 10))  # Corrected xticks range

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
