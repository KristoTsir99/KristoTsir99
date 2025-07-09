%% Step 4: Canonical Correlation Analysis

function [cca_Fmri, cca_Metabol] = run_cca(Fmri_PCA, Metabol_PCA, number_components)
    [A, B, ~] = canoncorr(Fmri_PCA, Metabol_PCA);
    cca_Fmri = Fmri_PCA * A(:, 1:number_components);
    cca_Metabol = Metabol_PCA * B(:, 1:number_components);
end