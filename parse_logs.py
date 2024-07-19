import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_log_files(log_dir, log_prefix, num_logs):
    # Initialize lists to store the final epoch metrics for each run
    final_train_loss = []
    final_val_loss = []
    final_train_acc = []
    final_val_acc = []
    test_accs = []
    test_losses = []

    for i in range(1, num_logs + 1):
        log_file = os.path.join(log_dir, f'{log_prefix}_{i}.txt')
        with open(log_file, 'r') as file:
            lines = file.readlines()
        
        # Initialize variables to store the values of the last epoch
        train_loss = val_loss = train_acc = val_acc = 0.0
        for line in lines:
            if line.startswith('Epoch'):
                # Extract the metrics using regular expressions
                match = re.match(r'Epoch \d+: loss=([0-9.]+), accuracy=([0-9.]+), val_loss=([0-9.]+), val_accuracy=([0-9.]+)', line)
                if match:
                    train_loss = float(match.group(1))
                    train_acc = float(match.group(2))
                    val_loss = float(match.group(3))
                    val_acc = float(match.group(4))
            elif line.startswith('Test accuracy'):
                test_acc = float(line.split(': ')[1].strip())
                test_accs.append(test_acc)
            elif line.startswith('Test loss'):
                test_loss = float(line.split(': ')[1].strip())
                test_losses.append(test_loss)
        
        # Append the final epoch values to the lists
        final_train_loss.append(train_loss)
        final_val_loss.append(val_loss)
        final_train_acc.append(train_acc)
        final_val_acc.append(val_acc)
    
    # Calculate the averages
    avg_train_loss = np.mean(final_train_loss)
    avg_val_loss = np.mean(final_val_loss)
    avg_train_acc = np.mean(final_train_acc)
    avg_val_acc = np.mean(final_val_acc)
    avg_test_acc = np.mean(test_accs)
    avg_test_loss = np.mean(test_losses)

    return {
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'avg_train_acc': avg_train_acc,
        'avg_val_acc': avg_val_acc,
        'avg_test_acc': avg_test_acc,
        'avg_test_loss': avg_test_loss
    }

def plot_metrics(metrics_unbalanced, metrics_balanced):
    labels = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Test Loss']
    unbalanced_values = [metrics_unbalanced['avg_train_loss'], metrics_unbalanced['avg_val_loss'], metrics_unbalanced['avg_train_acc'], metrics_unbalanced['avg_val_acc'], metrics_unbalanced['avg_test_acc'], metrics_unbalanced['avg_test_loss']]
    balanced_values = [metrics_balanced['avg_train_loss'], metrics_balanced['avg_val_loss'], metrics_balanced['avg_train_acc'], metrics_balanced['avg_val_acc'], metrics_balanced['avg_test_acc'], metrics_balanced['avg_test_loss']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, unbalanced_values, width, label='Unbalanced')
    rects2 = ax.bar(x + width/2, balanced_values, width, label='Balanced')

    ax.set_ylabel('Values')
    ax.set_title('Metrics by Unbalanced and Balanced Class Weights')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

log_dir_unbalanced = 'D:\\xray_classification_model\\Logs_unbalanced'
log_dir_balanced = 'D:\\xray_classification_model\\Logs_balanced'

log_prefix = 'training_log'
num_logs = 10

metrics_unbalanced = parse_log_files(log_dir_unbalanced, log_prefix, num_logs)
metrics_balanced = parse_log_files(log_dir_balanced, log_prefix, num_logs)

# Plot the metrics for comparison
plot_metrics(metrics_unbalanced, metrics_balanced)


