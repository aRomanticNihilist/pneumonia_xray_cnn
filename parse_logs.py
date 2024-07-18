import os
import re
import numpy as np

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

    # Print the averages
    print(f'Average Training Loss: {avg_train_loss}')
    print(f'Average Validation Loss: {avg_val_loss}')
    print(f'Average Training Accuracy: {avg_train_acc}')
    print(f'Average Validation Accuracy: {avg_val_acc}')
    print(f'Average Test Accuracy: {avg_test_acc}')
    print(f'Average Test Loss: {avg_test_loss}')

    return {
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'avg_train_acc': avg_train_acc,
        'avg_val_acc': avg_val_acc,
        'avg_test_acc': avg_test_acc,
        'avg_test_loss': avg_test_loss
    }

# Example usage
log_dir = 'D:\\xray_classification_model\\Logs_unbalanced'
log_prefix = 'training_log'
num_logs = 10

metrics = parse_log_files(log_dir, log_prefix, num_logs)
