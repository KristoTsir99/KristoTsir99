% The code below aims to perform proteomics data analysis (with artificial data), 
% pre-process them, statistically analyze them, and visualize the results.  
% The code was adapted from ChatGPT.
% For further information, you may visit https://openai.com/chatgpt


% First, we set a random seed for reproducibility
rng(50); % let's say 50

% Parameters for our artificial/synthetic proteomic data
number_proteins = 300;
number_samples_per_group = 40;

% Data
control_group = normrnd(40, 2, [number_proteins, number_samples_per_group]);
treatment_group = normrnd(40, 2, [number_proteins, number_samples_per_group]);

% Differential expression of 15 proteins
differential_expression_idx = randperm(number_proteins, 15);
treatment_group(differential_expression_idx, :) = treatment_group(differential_expression_idx, :) + normrnd(3, 0.5, [15 number_samples_per_group]);

% Combinining the data
data_combined = [control_group, treatment_group];
log2_data = log2(data_combined + 1);

% Pre=processing - Z-score normalization
data_z_score = (log2_data - mean(log2_data, 2)) ./ std(log2_data, 0, 2);

% Principal Component Analysis
[coeff, score, ~] = pca(log2_data');

% Plotting Principal Component Analysis
figure;
gscatter(score(:,1), score(:,2), [repmat({'Control'}, number_samples_per_group, 1); repmat({'Treatment'}, number_samples_per_group,1)]);
title('Principal Components');
xlabel('Component 1');
ylabel('Component 2');
grid on;

% Generating the heatmap of the 15 most variable proteins
variances = var(log2_data, 0, 2);
[~, idx] = maxk(variances, 15);
figure;
heatmap(log2_data(idx, :));
title('Top 15 Most Variable Proteins');

% Performing differential expression analysis
log2fc = mean(log2_data(:, number_samples_per_group+1:end), 2) - mean(log2_data(:, 1:number_samples_per_group), 2);
pvals = zeros(number_proteins, 1);

for i = 1:number_proteins
    [~, pvals(i)] = ttest2(log2_data(i, 1:number_samples_per_group), log2_data(i, number_samples_per_group+1:end));
end

% Adjusting p-values using Benjamini-Hochberg
[sorted_pvals, sort_idx] = sort(pvals);
adj_pvals = sorted_pvals .* number_proteins ./ (1:number_proteins)';
adj_pvals(adj_pvals > 1) = 1;
adj_pvals_original = zeros(size(adj_pvals));
adj_pvals_original(sort_idx) = adj_pvals;

% Generating a volcano plot
figure;
scatter(log2fc, -log10(pvals), 40, adj_pvals_original < 0.05, 'filled');
colormap([0 1 0; 1 1 0]); 
hold on;
yline(-log10(0.05), '--b');
xline(1, '--g');
xline(-1, '--g');
title('Volcano Plot');
xlabel('Log2 Fold Change');
ylabel('-Log10 p-value');
grid on;