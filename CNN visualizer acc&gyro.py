#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.signal import savgol_filter


# In[2]:


IMU_data = "# add your file path"
skip_rows = 11
columns = ['Timestamp', 'Accelerometer X', 'Accelerometer Y', 'Accelerometer Z', 'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z']

COI = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z', 'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z']
# Read data into Pandas DataFrame
df = pd.read_csv(IMU_data, sep=',', skiprows=skip_rows, header=None, names=columns)

print(df.head())


# In[3]:


# Convert Timestamp to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set Timestamp as index
df = df.set_index('Timestamp')


# In[8]:


# Apply Savitzky-Golay filter to accelerometer columns
df_filtered = df.copy()  

# Apply SG filter to each column
for column in COI:
    df_filtered[column] = savgol_filter(df[column], window_length=8001, polyorder=3)

# Display the filtered DataFrame
print(df_filtered.head(5))


# In[5]:


#Applying rolling window
IMU_SG_RW = df_filtered.rolling(window=50000).mean()
print(IMU_SG_RW.head())


# In[6]:


#Here is the overall volume of data that will be plotted. Want to visualize less? Then reduce the end time
start_time_str = '2024-01-26 16:00:00.000'
end_time_str = '2024-01-27 16:00:00.000'

start_time = pd.to_datetime(start_time_str)
end_time = pd.to_datetime(end_time_str)

IMU_selected = IMU_SG_RW.loc[start_time:end_time]


# In[7]:


# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

for column in IMU_selected.columns:
    sns.lineplot(data=IMU_selected, x=IMU_selected.index, y=column)

#Delete prior to use for CNN
plt.title(' Daily Accelerometer and Gyroscope Data')
plt.xlabel('Time (in hours)')
plt.ylabel('Displacement (in g-forces)')


tick_labels = pd.date_range(start=start_time, end=end_time, freq='H').strftime('%H')
plt.xticks(pd.date_range(start=start_time, end=end_time, freq='H'), tick_labels, rotation=45, ha='right')
plt.tight_layout()
plt.show()

