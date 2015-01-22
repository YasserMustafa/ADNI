function showConversionTime(labels, convtime, idx, path, K, name)

folds = numel(path);
mci.times = [];
nl.times = [];
mci.group = [];
nl.group = [];

MCI = 2;
NL = 1;

for fold=1:folds
    % stackedTime = cell2mat(convtime(idx{fold})')';
    % stackedLabels = cell2mat(labels(idx{fold})')';
    % stackedPath = cell2mat(path{fold}')';
    
    stackedTime = cellfun(@(seq)seq(:, end), convtime(idx{fold}));
    stackedLabels = cellfun(@(seq)seq(:, end), labels(idx{fold}));
    stackedPath = cellfun(@(seq)seq(:, end), path{fold});
        
    assert(isequal(size(stackedTime), ...
        size(stackedLabels), size(stackedPath)), ...
        'Size mismatch in conversion times');
    for k=1:K
        % relevant MCI times
        mciRel = stackedTime(stackedLabels==MCI & ...
            stackedPath==k & ...
            stackedTime~=-1);
        % relevant NL times
        nlRel = stackedTime(stackedLabels==NL & ...
            stackedPath==k & ...
            stackedTime~=-1);
        mci.times = [mci.times; mciRel];
        nl.times = [nl.times; nlRel];
        mci.group = [mci.group; ones(numel(mciRel), 1)*k];
        nl.group = [nl.group; ones(numel(nlRel), 1)*k];
    end
end

figure;

ax1 = subplot(2, 1, 1);
boxplot(mci.times, mci.group);
ylabel('Conversion Time (months)');
title(sprintf('%s - %s', 'MCI', name));

ax2 = subplot(2, 1, 2);
bar(1:K, histc(mci.group, 1:K));
xlabel('HMM State');
ylabel('Count');

linkaxes([ax1, ax2], 'x');

figure;

ax1 = subplot(2, 1, 1);
boxplot(nl.times, nl.group);
ylabel('Conversion Time (months)');
title(sprintf('%s - %s', 'NL', name));

ax2 = subplot(2, 1, 2);
bar(1:K, histc(nl.group, 1:K));
xlabel('HMM State');
ylabel('Count');

linkaxes([ax1, ax2], 'x');

end