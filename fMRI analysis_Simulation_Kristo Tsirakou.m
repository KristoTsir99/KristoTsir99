% The code below aims to simulate fMRI data and analyze it.
% The code was adapted from ChatGPT.
% For further information, you may visit https://openai.com/chatgpt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We first set the parameters of the simulated datasets
Shape = [32, 32, 18, 120]; % x,y,z coordinates and time series
noise_level = 0.5; 

% Then, we simulate the baseline noise
Fmri_data = noise_level * randn(Shape);

% We proceed by defining the activation region and timecourse
Activation_region_x = 12:19;
Activation_region_y = 12:19;
Activation_region_z = 6:9;
time_points = Shape(4);
Activation_timecourse = sin(linspace(0, 3*pi, time_points));

% Injecting activation over time
for t = 1:time_points
    Fmri_data(Activation_region_x, Activation_region_y, Activation_region_z, t) = ...
        Fmri_data(Activation_region_x, Activation_region_y, Activation_region_z, t) + Activation_timecourse(t);
end

% We apply Gaussian smoothing at each time point
sigma = 1;
for t = 1:time_points
    Fmri_data(:, :, :, t) = imgaussfilt3(Fmri_data(:, :, :, t), sigma);
end

% Visualizing the mean brain activation slice
z_slice = 8;
mean_brain = mean(Fmri_data, 4);

figure('Position', [100 100 1200 400]);

subplot(1,2,1);
imagesc(mean_brain(:, :, z_slice));
axis image; colormap(gray); colorbar;
title(['Mean activation (z=' num2str(z_slice) ')']);

% Voxel time series
Voxel_coords = [16, 16, z_slice];
Voxel_ts = squeeze(Fmri_data(Voxel_coords(1), Voxel_coords(2), Voxel_coords(3), :));

subplot(1,2,2);
plot(Voxel_ts, 'LineWidth', 1.5);
title(['Voxel Time Series @ (' num2str(Voxel_coords) ')']);
xlabel('Time');
ylabel('Signal');

% Computing voxel-wise correlation with the activation regressor
Design_regressor = zscore(Activation_timecourse);
Correlation_map = zeros(Shape(1:3));

for x = 1:Shape(1)
    for y = 1:Shape(2)
        for z = 1:Shape(3)
            ts = squeeze(Fmri_data(x,y,z,:));
            ts_z = zscore(ts);
            r = corrcoef(ts_z, Design_regressor);
            Correlation_map(x,y,z) = r(1,2);
        end
    end
end

% Thresholding
Threshold = 0.5;
Activation_mask = Correlation_map > Threshold;
number_active_voxels = sum(Activation_mask(:));

% Visualizing of 2D results
figure('Position', [100 100 1200 500]);

subplot(1,2,1);
imagesc(Correlation_map(:, :, z_slice));
colormap(hot);
caxis([0 1]);
colorbar;
axis image;
title(['Correlation Map (z=' num2str(z_slice) ')']);

subplot(1,2,2);
imagesc(Activation_mask(:, :, z_slice));
colormap(gray);
axis image;
title(['Activation Mask (r > ' num2str(Threshold) ')']);

% Reporting the activated voxels
fprintf('Total activated voxels (r > %.1f): %d\n', Threshold, number_active_voxels);


% 3D Visualization and rotation of activation mask

figure('Name','3D Brain Activation Mask','Color','w')
p = patch(isosurface(Activation_mask, 0.5));
isonormals(Activation_mask, p)
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1])
view(3)
camlight; lighting gouraud
axis tight
grid on
xlabel('X'); ylabel('Y'); zlabel('Z');
title(['3D Brain Activation Mask (r > ' num2str(Threshold) ')']);
alpha(0.7)

% Rotating the 3D plot continuously
for az = 0:2:360
    view(az, 30)
    pause(0.05)
end
