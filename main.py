import time
from collections import defaultdict
import queue
import threading
import pandas as pd


import hydra
import torch
from tqdm import tqdm
from omegaconf import DictConfig

from src import utils


def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)

    for epoch in tqdm(range(opt.training.epochs)):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, "val", epoch=epoch)

    return model


def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    validate_or_test(opt, model, "val")

    if opt.training.final_test:
        validate_or_test(opt, model, "test")


if __name__ == "__main__":
    
   
    ## Define logging queues
    power_log_queue = queue.Queue()
    util_log_queue = queue.Queue()
    memory_log_queue = queue.Queue()

    ## Define stop event
    stop_event = threading.Event()
    event_start = time.time() # for plotting timestamps


    ## Define logging threads
    power_logging_thread = threading.Thread(
        target=utils.log_gpu_power, args=(stop_event, power_log_queue)
    )
    util_logging_thread = threading.Thread(
        target=utils.log_gpu_util, args=(stop_event, util_log_queue)
    )
    mem_logging_thread = threading.Thread(
        target=utils.log_gpu_mem, args=(stop_event, memory_log_queue)
    )

    ## Start logging threads
    power_logging_thread.start()
    util_logging_thread.start()
    mem_logging_thread.start()
    
    FF_start_time = time.time()
    my_main()
    FF_end_time = time.time()
    
    stop_event.set()
    mem_logging_thread.join()
    util_logging_thread.join()
    power_logging_thread.join()

    ## Extracting logs
    power_log = []
    while not power_log_queue.empty():
        power_log.append(power_log_queue.get())

    util_log = []
    while not util_log_queue.empty():
        util_log.append(util_log_queue.get())

    memory_log = []
    while not memory_log_queue.empty():
        memory_log.append(memory_log_queue.get())

    ## Extracting timestamps
    power_timestamps = [x[0] for x in power_log]
    power_values = [x[1] for x in power_log]

    util_timestamps = [x[0] for x in util_log]
    util_values = [x[1] for x in util_log]

    memory_timestamps = [x[0] for x in memory_log]
    memory_values = [x[1] for x in memory_log] # convert to MB
    
    ## Saving log data as CSV
    pd.DataFrame(power_log, columns=["Timestamp", "Value"]).to_csv("./Outputs/power_log.csv" , index = False)
    pd.DataFrame(util_log, columns=["Timestamp", "Value"]).to_csv("./Outputs/util_log.csv", index=False)
    pd.DataFrame(memory_log, columns=["Timestamp", "Value"]).to_csv("./Outputs/memory_log.csv", index=False)
    
    ## Saving FF and BP timestamps as CSV
    model_timestamps = {"FF": [FF_start_time, FF_end_time]}
    pd.DataFrame(model_timestamps).to_csv("./Outputs/model_timestamps.csv", index = False)

    ## Function to print the power log
    plt.plot(power_timestamps, power_values)
    plt.title('Power Draw Comparison')
    plt.xlabel('Time (sec)')
    plt.ylabel('Power')
    plt.axvline(x=FF_start_time, color="g", linestyle="--", label="FF Start")
    plt.axvline(x=FF_end_time, color="g", linestyle="--", label="FF End")
    plt.legend()
    plt.savefig("./Images/power_log.png")
    plt.clf()

    ## Function to print the utilization log
    plt.plot(util_timestamps,util_values)
    plt.title('Memory Utilisation')
    plt.xlabel('Time (sec)')
    plt.ylabel('Memory Utilisation (%)')
    plt.axvline(x=FF_start_time, color="g", linestyle="--", label="FF Start")
    plt.axvline(x=FF_end_time, color="g", linestyle="--", label="FF End")
    plt.legend()
    plt.savefig("./Images/util_log.png")
    plt.clf()
    
    ## Function to print the memory log
    plt.plot(memory_timestamps, memory_values)
    plt.title('Memory Usage')
    plt.xlabel('Time (sec)')
    plt.ylabel('Memory Usage (MB)')
    plt.axvline(x=FF_start_time, color="g", linestyle="--", label="FF Start")
    plt.axvline(x=FF_end_time, color="g", linestyle="--", label="FF End")
    plt.legend()
    plt.savefig("./Images/memory_log.png")
    plt.clf()



