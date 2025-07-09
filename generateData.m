% The code below aims to simulate fMRI and metabolomic data and analyze them
% by performing pre-processing and visualization. The main objective is to
% extract meaningful information about the brain connectivity patterns and 
% the molecular profiling of various diseases,
% let's say Alzheimer's or Schizophrenia.  

% The code was adapted from ChatGPT.
% For further information, you may visit https://openai.com/chatgpt

%% Step 1: Simulating the fMRI and Metabolomic data

function [Fmri_data, Metabol_data, labels] = generateData(number_subjects, number_regions, number_metabolites, disease_ratio)
    rng(55); % we set a random seed for reproducibility

    labels = randsample([0 1], number_subjects, true, [1 - disease_ratio, disease_ratio]);

    number_features = number_regions * (number_regions - 1) / 2;
    Fmri_data = zeros(number_subjects, number_features);
    Metabol_data = zeros(number_subjects, number_metabolites);

    for i = 1:number_subjects
        label = labels(i);
        base_con = rand(number_regions);
        con_matrix = (base_con + base_con') / 2;

        if label == 1
            con_matrix = con_matrix + 0.2 * eye(number_regions);
        else
            con_matrix = con_matrix + 0.1 * eye(number_regions);
        end

        upper_index = find(triu(ones(number_regions), 1));
        Fmri_data(i, :) = con_matrix(upper_index);

        base_metabol = randn(1, number_metabolites);
        if label == 1
            base_metabol(1:5) = base_metabol(1:5) + 2;
        end
        Metabol_data(i, :) = base_metabol;
    end
end