import mne
import os

# Path to GDF file
gdf_path = os.path.join("data", "A01T.gdf")

# Load the raw GDF EEG file
raw = mne.io.read_raw_gdf(gdf_path, preload=True, stim_channel='auto')

# Print basic info
print(raw.info)

# Plot the EEG (10 channels, first 5 seconds)
raw.plot(start=0, duration=5, n_channels=10, scalings='auto')

# Save the data shape
data, times = raw[:22, :]  # 22 EEG channels
print("Data shape:", data.shape)  # (channels, time samples)



# to activate the environment: source neuro_env/bin/activate