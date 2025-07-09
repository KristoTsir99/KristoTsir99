%% Plotting CCA scatter plots
function plot_cca_scatter(cca_Fmri, cca_Metabol, labels)
    figure;
    subplot(1,2,1);
    gscatter(cca_Fmri(:,1), cca_Fmri(:,2), labels);
    title('CCA - fMRI');

    subplot(1,2,2);
    gscatter(cca_Metabol(:,1), cca_Metabol(:,2), labels);
    title('CCA - Metabolomics');
end