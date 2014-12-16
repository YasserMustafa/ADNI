function [counts, yy] = showStateClinicalDist(clinVar, idx, path, K, name)
%% Plot the distribution of clinical variable in each state of the HMM
%
% clinVar           -- vector of the continuous clinical variable
% idx               -- idx{i} has the training indices for fold i
% path              -- path{i} has the Viterbi paths for fold i
% K                 -- number of HMM states
% name              -- name of the plot

%%

folds = size(path, 2);
yy = unique(cell2mat(clinVar')');
counts = zeros(numel(yy), K, folds);

for i=1:folds
    clin = cell2mat(clinVar(idx{i})')';
    vit = cell2mat(path{i}')';
    for k=1:K
        dist = clin(vit==k);
        counts(:, k, i) = histc(dist, yy);
    end
    counts(:, :, i) = normalize(counts(:, :, i), 1);
end

scores = sum(counts.*repmat(yy, 1, size(counts, 2), size(counts, 3)), 1);
avg = mean(scores, 3);
stdev = std(scores, 0, 3);
counts = mean(counts, 3);

figure;
imagesc(counts);
ax = gca;
ax.YTick = 1:size(counts, 1);
ax.YTickLabel = yy;
ax.XTick = 1:size(counts, 2);
axis image;
xlabel('HMM State');
ylabel('Clinical Score');
title(name);
colorbar;
colormap hot;

figure;
errorbar(1:K, avg, stdev);
xlabel('HMM State');
ylabel('Mean Clinical Score');
title(name);

end