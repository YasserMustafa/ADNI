function [model, ll] = pet_hmm()

close all;
set(0, 'DefaultAxesFontSize', 20);
set(0, 'DefaultTextFontSize', 20);

% read in and clean the FDG-PET scan data, and condition it to be used by
% the pmtk3 package

data_pet = '/phobos/alzheimers/adni/pet_flat.csv';

% must use this because there is some non-numeric data in table (e.g.
% VISCODE2)
data = readtable(data_pet);

% grab the DX for each patient
labels = data.DX;
nl = cellfun(@strcmp, labels, repmat({'NL'}, size(labels, 1), 1));
mci = cellfun(@strcmp, labels, repmat({'MCI'}, size(labels, 1), 1));
ad = cellfun(@strcmp, labels, repmat({'AD'}, size(labels, 1), 1));
labels = zeros(size(labels, 1), 1);
labels(nl) = 1;
labels(mci) = 2;
labels(ad) = 3;

% ignore VISCODE2 for now, which is column #2
%data = table2array(data(:, [1, 3:end]));
% the high dimensionality of the data is causing singular problems with the
% covariance matrices. Just work with means for now.
data = table2array(data(:, [1, 4:6:end]));

% divide up the data by RIDs
% RID = data(:, 1)
[~, order] = sort(data(:, 1)); % first sort based on RID
data = data(order, :); % make sure the RIDs appear in ascending order
% generate counts for each RID
counts = histc(data(:, 1), unique(data(:, 1))); 
rid = data(:, 1);

stackedData = data(:, 2:end);
stackedLabels = labels;

% counts will have the number of rows that belong to each RID
% generate a cell array now, using these counts to divide up the matrix
data = mat2cell(data(:, 2:end), counts);
labels = mat2cell(labels, counts);
% make each column an obervation rather than each row
data = cellfun(@transpose, data, 'UniformOutput', false);
labels = cellfun(@transpose, labels, 'UniformOutput', false);

dx = rowvec(sort(unique(stackedLabels)));
gt.pi = histc(stackedLabels, dx)/numel(stackedLabels)';
gt.A = zeros(numel(dx));
trans = cellfun(@countTransitions, labels, ...
                repmat({dx}, numel(labels), 1), 'UniformOutput', false);
gt.A = sum(cat(3, trans{:}), 3);
gt.A = bsxfun(@rdivide, gt.A, sum(gt.A, 2));

% divide the data into training and testing sets
idx = randperm(numel(data));
train_data = data(idx(1:1000));
train_lab = labels(idx(1:1000));
test_data = data(idx(1000:end));
test_lab = labels(idx(1000:end));

[model, ll] = hmmFit(train_data, 5, 'gauss', 'verbose', true, ...
                    'maxIter', 100, 'nRandomRestarts', 3);

[~, train_conv, train_path] = plotTrellis(train_data, train_lab, model, 'Training data');
[~, test_conv, test_path] = plotTrellis(test_data, test_lab, model, 'Testing data');

train_lab = cell2mat(train_lab');
train_path = cell2mat(train_path');

test_lab = cell2mat(test_lab');
test_path = cell2mat(test_path');

K = numel(model.pi);
Y = 3;
labNames = {'NL', 'MCI', 'AD'};

plotViterbi(train_lab, train_path, Y, K, labNames, 'Training data');
plotViterbi(test_lab, test_path, Y, K, labNames, 'Testing data');

end

function plotViterbi(labels, path, Y, K, labNames, name)

edges = 1:K;

figure;
hold on;

h = zeros(Y, K);
l = cell(1, Y);

for lab=1:Y
    h(lab, :) = hist(path(labels==lab), edges);
    l{lab} = labNames{lab};
end

plt = bar(edges, h', 0.4, 'grouped');
grid on;
legend(plt, l);
title(name);
xlabel('HMM State number')
ylabel('Count')

hold off;

end

function [trellis, conv, path] = plotTrellis(data, labels, model, name)

MCI = 2;
AD = 3;

path = cell(size(data)); % hold the viterbi path for every observation seq.
t_max = max(cellfun(@(x)size(x, 2), data)); % the longest obs. seq. over all patients
k = numel(model.pi); % number of states

% store the number of transitions from every possible source to every
% possible destination. The row represents the source, and the column the 
% destination state. The matrix in cell {t} represents the transitions seen
% at time = t.
trellis = repmat({zeros(k)}, t_max-1, 1);
% store information about the conversion from MCI to AD. For every
% conversion from MCI to AD seen in the gt, mark the corresponding
% conversion in the HMM state space in the following matrix. Again, rows
% are sources and cols are destination states. 
conv = repmat({zeros(k)}, t_max-1, 1);

for i=1:numel(data)
    path{i} = hmmMap(model, data{i});
    for t=1:numel(path{i})-1
        assert(numel(path{i})==numel(labels{i}), ...
            'Labels and data sync mismatch');
        r = path{i}(t);
        c = path{i}(t+1);
        trellis{t}(r, c) = trellis{t}(r, c) + 1;
        
        if labels{i}(t) == MCI && labels{i}(t+1) == AD
            conv{t}(r, c) = conv{t}(r, c) + 1;
        end
    end
end

figure;
hold on;
xlim([0, t_max+1])
ylim([0, k+1])
xlabel('Time')
ylabel('HMM State')
max_wt = max(cellfun(@(x)max(x(:)), trellis));

for t=1:numel(trellis)
    [r, c] = find(trellis{t});
    for i=1:numel(r)
        plot([t, t+1], [r(i), c(i)], ...
            'Marker', 'x', 'MarkerSize', 10, ...
            'LineStyle', '-', 'LineWidth', 25*trellis{t}(r(i), c(i))/max_wt);
    end
end
hold off;

title(name);

end

function trans = countTransitions(labels, dx)

trans = zeros(numel(dx));
for i=1:numel(dx)
    t1 = find(labels==dx(i));
    t2 = t1 + 1;
    t2 = t2(t2 <= numel(labels));
    t1 = t1(1:numel(t2));
    if ~isempty(t1) && ~isempty(t2)
        idx = @(src, dest)sum(labels(t1)==src ...
            & labels(t2)==dest);
        trans(i, :) = arrayfun(idx, repmat(dx(i), 1, numel(dx)), dx);
    end
end

end



