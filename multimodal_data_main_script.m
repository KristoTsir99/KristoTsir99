% Parameters
number_subjects = 300;
number_regions = 50;
number_metabolites = 20;
disease_ratio = 0.5;
number_components = 10;

% Simulate Data
[Fmri_data, Metabol_data, labels] = generateData(number_subjects, number_regions, number_metabolites, disease_ratio);

% Preprocessing
[Fmri_scaled, Metabol_scaled] = data_preprocessed(Fmri_data, Metabol_data);

% PCA
[Fmri_PCA, Metabol_PCA] = reduce_dimensions(Fmri_scaled, Metabol_scaled, number_components);

% CCA
[cca_Fmri, cca_Metabol] = run_cca(Fmri_PCA, Metabol_PCA, number_components);

% Classification
classify_multimodal_data(cca_Fmri, cca_Metabol, labels);

% Plot
plot_cca_scatter(cca_Fmri, cca_Metabol, labels);

% Connectivity plots
average_healthy = mean(Fmri_data(labels == 0, :), 1);
average_diseased = mean(Fmri_data(labels == 1, :), 1);
plot_connectivity_graph(average_healthy, number_regions, 'Connectivity - Healthy');
plot_connectivity_graph(average_diseased, number_regions, 'Connectivity - Diseased');