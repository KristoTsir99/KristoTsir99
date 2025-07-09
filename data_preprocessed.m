%% Step 2: Data Pre-processing

function [Fmri_scaled, Metabol_scaled] = data_preprocessed(Fmri_data, Metabol_data)
    Fmri_scaled = zscore(Fmri_data);
    Metabol_scaled = zscore(Metabol_data);
end