%% Plotting Connectivity Graphs

function plot_connectivity_graph(average_vector, number_regions, title_str)
    con_matrix = zeros(number_regions);
    upper_index = find(triu(ones(number_regions), 1));
    con_matrix(upper_index) = average_vector;
    con_matrix = con_matrix + con_matrix';

    G = graph(con_matrix);
    figure;
    plot(G, 'Layout', 'force', 'NodeColor', 'r', 'EdgeAlpha', 0.4);
    title(title_str);
end