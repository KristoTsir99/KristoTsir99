% This code is a matlab version of the r code conducted on NLP. 
% The code was adapted from ChatGPT
% For further information, you may visit https://openai.com/chatgpt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sample texts
text_1 = 'Looking at the sky, the ocean seems to cry, nature wanna run';
text_2 = 'Joy closer coming, magic never stopping, sky above us shining';

% Storing texts into one variable
sample_texts = {text_1, text_2}; % 1X2 cell array

% Conversion of the texts into lowercase
text_1 = lower(text_1);
text_2 = lower(text_2);

% Punctuation removal
text_1 = regexprep(text_1, '[^\w\s]', '');
text_2 = regexprep(text_2, '[^\w\s]', '');

% Text tokenization
tokens_text_1 = strsplit(text_1);
tokens_text_2 = strsplit(text_2);

% Combining both tokens
tokens_combined = unique([tokens_text_1, tokens_text_2]);

% Term-document matrix (DTM)
DTM = ones(length(tokens_combined), 2);

% DTM creation
for i = 1:length(tokens_combined)
    DTM(i, 1) = sum(strcmp(tokens_text_1, tokens_combined{i}));
    DTM(i, 2) = sum(strcmp(tokens_text_2, tokens_combined{i}));
end

% Display the term-document matrix
disp('Term-Document Matrix:');
disp(DTM);

% Cosine similarity
dot_text_product = dot(DTM(:, 1), DTM(:, 2));
norm_text_1 = norm(DTM(:, 1));  
norm_text_2 = norm(DTM(:, 2));  

cosine_similarity = dot_text_product / (norm_text_1 * norm_text_2);

% Displaying the cosine similarity
disp(['Cosine Similarity between Text 1 and Text 2: ', num2str(cosine_similarity)]);

% Create a similarity matrix for visualization
similarity_matrix = [1, cosine_similarity; cosine_similarity, 1];

% Plotting the heat-map of the similarity matrix
figure;
imagesc(similarity_matrix);
colorbar;
title('Text Similarity Matrix');
xticks([1 2]);
xticklabels({'Text 1', 'Text 2'});
yticks([1 2]);
yticklabels({'Text 1', 'Text 2'});

% Indexing the sample texts
text1Index = 1;
text2Index = 2; 

% Creating a bar-plot of the similarity matrix
bar(similarity_matrix);
xlabel('Text Index');
ylabel('Cosine Similarity');
title(['Cosine Similarity to Text', num2str(text1Index)]);
ylim([0 1]);

bar(similarity_matrix);
xlabel('Text Index');
ylabel('Cosine Similarity');
title(['Cosine Similarity to Text', num2str(text2Index)]);
ylim([0 1]); 

%%%%%%%%%%%%%%%%% End of NLP coding %%%%%%%%%%%%%%%%%%%%%%