%% Example of LDA
load data_D2_C2.mat

%% Accesing Data
pvalues = p.value;
plabels = p.class;

%% Normalizing data
disp("-------- Normalizing Data ---------------------------");
disp("Normalizing data...");
[Mp, Np] = size(pvalues);
mn_p = mean(pvalues')';
std_p = std(pvalues')';
for i = 1:Np
    pn(:,i) = (pvalues(:,i) - mn_p)./std_p;
end

%% Plotting Normalized Data vs Raw Data
% one_plot('Raw Data vs Centred Data', 'p.values', ...
%             'x - coordinates', 'y - coordinates', 'Raw Data', ...
%             'Centered Data', pvalues, pn, ...
%             plabels, 'centred_data_vs_raw_data.png');

%% Computing Scatter Matrices
pmatrices = Scatter_matrices(pn, plabels);
S1 = pmatrices{1};
S2 = pmatrices{2};
Sw = pmatrices{3}
Sb = pmatrices{4}
St = pmatrices{5};

%% Computing W matrix: maximizing the Fisher Discriminant
[W, D] = eig(inv(Sw)*Sb)

%% Sort the variances in decreasing order
disp("-------- Sorting variances in decreasing order ---------");
% Extract diagonal of matrix as vector
D = diag(D);
% Sort W and convert D to a column vector with the eigenvalues
[~, p_rindices] = sort(-1*D);
D = D(p_rindices);
W = W(:, p_rindices);


%% Computing projection
w = W(:,1)';
y = w * pn;

%% Desprojection
x_n = w' * y;
for i=1:Np
    x(:,i) = x_n(:,i) .* std_p + mn_p;
end

%% Plotting Data vs LDA projection
% one_plot('Raw Data vs LDA Projection', ...
%             'p.values', 'x - coordinates', ...
%             'y - coordinates', ...
%           'Raw Data', 'Projected Data', pvalues, x, ...
%             plabels, 'lda_plot_first.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Auxiliar Functions 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function one_plot(generic_title, title_subplot_1, ...
                    x_1_label, y_1_label, ...
                    legend_1_light, legend_1, ...
                    light_data_1, data_1, ...
                    labels_1, saved_name)
    disp("-------- Plotting ----------------------------------");
    disp(generic_title);
    figure;
    % Generic Title
    sgtitle(generic_title);
    % First Subplot
    subplot(1, 1, 1);
    for i=1:length(labels_1)
        if labels_1(i) == 1
            scatter(light_data_1(1,i), light_data_1(2,i), 50, 'o', ...
                    'MarkerEdgeColor', 'none', 'MarkerFaceColor', [0 0.4470 0.7410], ...
                    'MarkerFaceAlpha', 0.1); hold on;
            scatter(data_1(1,i), data_1(2,i), 50, 'o', ...
                    'MarkerEdgeColor', 'none', 'MarkerFaceColor', [0 0.4470 0.7410], ...
                    'MarkerFaceAlpha', 1); hold on;
        else
            scatter(light_data_1(1,i), light_data_1(2,i), 50, 'o', ...
                    'MarkerEdgeColor', 'none', 'MarkerFaceColor', [0.8500 0.3250 0.0980], ...
                    'MarkerFaceAlpha', 0.1); hold on;
            scatter(data_1(1,i), data_1(2,i), 50, 'o', ...
                    'MarkerEdgeColor', 'none', 'MarkerFaceColor', [0.8500 0.3250 0.0980], ...
                    'MarkerFaceAlpha', 1); hold on;
        end
    end
    % Plot vertical lines for x=0 and y=0
    plot([0 0], ylim, 'k-');
    plot(xlim, [0 0], 'k-');
    % Subplot title
    title(title_subplot_1);
    % Axis labels
    xlabel(x_1_label);
    ylabel(y_1_label);
    legend({legend_1_light, legend_1}, 'Location', 'best');
    pbaspect([1 1 1]);
    % pos = get(gcf, 'Position');
    % set(gcf, 'Position',pos+[-900 -300 900 300])
    saveas(gca, saved_name);
end

function matrix_cell = Scatter_matrices(values, labels)
        % first we compute the number of labels
        unique_labels = unique(labels);
        N = length(unique_labels);
        % now we create as many vectors as labels: with cell arrays
        vectors_cell = cell(1, N);
        matrix_cell = cell (1, N+3);

        for i = 1:N
            label = unique_labels(i);
            indices = labels == label; % Find indices corresponding to the current label
            vectors_cell{i} = values(:, indices); % Store coordinates associated with the label
        end
        % Scatter matrix for each group is computed
        for i = 1:N
            data = vectors_cell{i};
            Sc = cov(data')*(length(data)-1);
            matrix_cell{i} = Sc;
        end
        % Scatter matrix within the groups is computed
        Sw = zeros(N);
        for i = 1:N
            Sw = Sw + matrix_cell{i};
        end
        s = N + 1;
        matrix_cell{s} = Sw;
        % Scatter matrix between the groups is computed
        Sb = zeros(N);
        m_x = mean(values(1,:));
        m_y = mean(values(2,:));
        m = [m_x; m_y];
        for i = 1:N
            data = vectors_cell{i};
            m_c_x = mean(data(1,:));
            m_c_y = mean(data(2,:));
            m_c = [m_c_x; m_c_y];
            Sb = Sb + length(data)*(m_c-m)*(m_c-m).';
        end
        s = s + 1;
        matrix_cell{s} = Sb;
        % Total Scatter matrix is computed
        St = (values - m)*(values - m).';
        s = s + 1;
        matrix_cell{s} = St;
end

