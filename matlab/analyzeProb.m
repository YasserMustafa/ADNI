function analyzeProb(prob, path, labels, idx, Y, name)
%% Analyze the Viterbi prob. matrix

%%

folds = size(prob, 2);
K = size(prob{1}{1}, 1);
avg_vit = zeros(K, folds);
avg_lab = zeros(Y, folds);

for f=1:folds
    vit = cell2mat(prob{f}')';
    seq = cell2mat(path{f}')';
    lab = cell2mat(labels(idx{f})')';
    for y=1:Y
        avg_lab(y, f) = mean(max(vit(lab==y, :), [], 2));
    end
    for k=1:K
        avg_vit(k, f) = mean(max(vit(seq==k, :), [], 2));
    end
end

figure;
hold on;
errorbar(1:Y, mean(avg_lab, 2), std(avg_lab, 0, 2));
errorbar(1:K, mean(avg_vit, 2), std(avg_vit, 0, 2));
legend('Clinical Labels', 'Viterbi State');
xlabel('State/Label Index');
ylabel('Mean probability');
title(name);

end