# The code below aims to simulate fMRI data and analyze it.
# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

######################################################################

# First, we import the required libraries and packages

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_stat_map
from nilearn.image import new_img_like
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore

# Then, we make a simulation of 4-dimension fMRI data
Shape = (32, 32, 17, 100)  #x,y,z dimensions and time
Fmri_data = np.random.randn(*Shape) * 0.5  # baseline noise correction


# We then define an activation region of our choice and our preferred timecourse
Activation_region = (slice(12, 20), slice(12, 20), slice(6, 10))
Activation_timecourse = np.sin(np.linspace(0, 3 * np.pi, Shape[3]))

# We inject the activation region over time
for t in range(Shape[3]):
    Fmri_data[Activation_region[0], Activation_region[1], Activation_region[2], t] += Activation_timecourse[t]

# We then apply Gaussian smoothing for spatial preprocessing simulation
for t in range(Shape[3]):
    Fmri_data[..., t] = gaussian_filter(Fmri_data[..., t], sigma=1)

# Visualization
z_slice = 8
mean_brain = np.mean(Fmri_data, axis=3)

plt.figure(figsize=(12, 4))

# For vizualizing the mean activation slice
plt.subplot(1, 2, 1)
plt.imshow(mean_brain[:, :, z_slice], cmap='gray')
plt.title("Mean activation (z=8)")
plt.colorbar()

# Voxel time series
Voxel_coords = (16, 16, z_slice)
Voxel_ts = Fmri_data[Voxel_coords[0], Voxel_coords[1], Voxel_coords[2], :]

plt.subplot(1, 2, 2)
plt.plot(Voxel_ts)
plt.title(f"Voxel Time Series @ {Voxel_coords}")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.tight_layout()
plt.show()

# Statistics
Design_regressor = zscore(Activation_timecourse)
Correlation_map = np.zeros(Shape[:3])

# Voxel-wise correlation with regressor
for x in range(Shape[0]):
    for y in range(Shape[1]):
        for z in range(Shape[2]):
            ts = zscore(Fmri_data[x, y, z, :])
            Correlation_map[x, y, z] = np.corrcoef(ts, Design_regressor)[0, 1]

# Thresholding
Threshold = 0.5
Activation_mask = Correlation_map > Threshold
number_active_voxels = np.sum(Activation_mask)

# Visualizing the results
plt.figure(figsize=(12, 5))

# Correlation map
plt.subplot(1, 2, 1)
plt.imshow(Correlation_map[:, :, z_slice], cmap='hot', vmin=0, vmax=1)
plt.title("Correlation Map (z=8)")
plt.colorbar(label='Correlation')

# Thresholded activation mask
plt.subplot(1, 2, 2)
plt.imshow(Activation_mask[:, :, z_slice], cmap='gray')
plt.title(f"Activation Mask (r > {Threshold})")
plt.tight_layout()
plt.show()

# Reporting
print(f"✅ Total activated voxels (r > {Threshold}): {number_active_voxels}")






