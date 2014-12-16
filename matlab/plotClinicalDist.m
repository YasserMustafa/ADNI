function plotClinicalDist(labels, clinVar, labNames, name)
%% Plot the distribution of the clinical variable at each clinical label
%
% Use this script to see dist. of MMSE/CDR at each of NL/MCI-C/MCI-NC/AD
% labels            -- vector of clinical labels 
% clinVar           -- vector of continuous clinical label
% labNames          -- name of each label for the legend
% name              -- name of the plot

%%
if iscell(labels)
    labels = cell2mat(labels')';
end

if iscell(clinVar)
    clinVar = cell2mat(clinVar')';
end

idx = clinVar ~= -1 & ~isnan(clinVar);
clinVar = clinVar(idx);
labels = labels(idx);

Y = numel(labNames);
uniq = unique(clinVar);
counts = zeros(numel(uniq), Y);

figure;
hold on;

for i=1:Y
    %dist = clinVar(labels==i);
    %uniq = unique(dist);
    %scatter(repmat(i, numel(uniq), 1), uniq, histc(dist, uniq));
    counts(:, i) = histc(clinVar(labels==i), uniq);
end

bar(uniq, counts, 'grouped');
legend(labNames);
xlabel('Score');
ylabel('Count');
title(name);
ax = gca;
ax.YGrid = 'on';

end