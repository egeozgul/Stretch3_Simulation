import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data from your specific path
data1 = pd.read_csv('scripts/warehouse_graphs/wandb_export_2024-09-25T10_08_39.946-04_00.csv')
data2 = pd.read_csv('scripts/warehouse_graphs/wandb_export_2024-09-25T10_09_01.989-04_00.csv')

# Plot first dataset
plt.plot(data1['Step'], data1['Grouped runs - mean_return'], label='Mac-IAICC', color='blue')
plt.fill_between(data1['Step'], data1['Grouped runs - mean_return__MIN'], 
                 data1['Grouped runs - mean_return__MAX'], color='blue', alpha=0.2)

# Plot second dataset
plt.plot(data2['Step'], data2['Grouped runs - mean_return'], label='Mac-IPPO', color='green')
plt.fill_between(data2['Step'], data2['Grouped runs - mean_return__MIN'], 
                 data2['Grouped runs - mean_return__MAX'], color='green', alpha=0.2)

# Add labels, title, and legend
plt.xlabel('Step')
plt.ylabel('Mean Return')
plt.title('Warehouse B')
plt.legend()

# Save the figure to a file instead of showing it
plt.savefig('/home/willy/macro_marl_ppo/scripts/warehouse_graphs/warehouse_b_comparison_plot.png')

# Optionally close the plot to free memory
plt.close()