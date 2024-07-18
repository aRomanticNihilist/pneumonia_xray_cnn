import pandas as pd
import matplotlib.pyplot as plt
import glob

# Load all log files into a DataFrame
log_files = glob.glob('log_bn_*.csv')
all_logs = []

for log_file in log_files:
    df = pd.read_csv(log_file)
    df['log_file'] = log_file
    all_logs.append(df)

logs_df = pd.concat(all_logs)

# Extract information
logs_df['batch_normalization'] = logs_df['log_file'].apply(lambda x: 'with' if 'True' in x else 'without')
logs_df['run'] = logs_df['log_file'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))

# Plot the metrics
plt.figure(figsize=(12, 6))

for key, grp in logs_df.groupby(['batch_normalization']):
    plt.plot(grp['epoch'], grp['val_accuracy'], label=f'Validation Accuracy ({key})')
    plt.plot(grp['epoch'], grp['val_loss'], label=f'Validation Loss ({key})')

plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Validation Accuracy and Loss over Epochs')
plt.legend()
plt.show()
