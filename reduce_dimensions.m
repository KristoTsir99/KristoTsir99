%% Step 3: Pricnipal Component Analysis (PCA)

function [Fmri_PCA, Metabol_PCA] = reduce_dimensions(Fmri_scaled, Metabol_scaled, number_components)
    [coeff1, score1] = pca(Fmri_scaled);
    [coeff2, score2] = pca(Metabol_scaled);
    Fmri_PCA = score1(:, 1:number_components);
    Metabol_PCA = score2(:, 1:number_components);
end