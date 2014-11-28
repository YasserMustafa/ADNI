function [model, ll] = pet_hmm()
%% Train a HMM model on the PET data, and visualize the results

%% Read and clean data

% read in and clean the FDG-PET scan data, and condition it to be used by
% the pmtk3 package

data_pet = '/phobos/alzheimers/adni/pet_flat.csv';

% must use this because there is some non-numeric data in table (e.g.
% VISCODE2)
pet = readtable(data_pet);

% grab the DX for each patient
labels = pet.DX;
nl = cellfun(@strcmp, labels, repmat({'NL'}, size(labels, 1), 1));
mci = cellfun(@strcmp, labels, repmat({'MCI'}, size(labels, 1), 1));
ad = cellfun(@strcmp, labels, repmat({'AD'}, size(labels, 1), 1));
labels = zeros(size(labels, 1), 1);
labels(nl) = 1;
labels(mci) = 2;
labels(ad) = 3;

% ignore VISCODE2 for now, which is column #2
% the high dimensionality of the data is causing singular problems with the
% covariance matrices. Just work with means for now.
pet = table2array(pet(:, [1, 4:6:end]));

% divide up the data by RIDs
% RID = data(:, 1)
[~, order] = sort(pet(:, 1)); % first sort based on RID
pet = pet(order, :); % make sure the RIDs appear in ascending order
% generate counts for each RID
counts = histc(pet(:, 1), unique(pet(:, 1))); 
%rid = data(:, 1);

%stackedData = data(:, 2:end);
stackedLabels = labels;

% counts will have the number of rows that belong to each RID
% generate a cell array now, using these counts to divide up the matrix
% each cell of this cell array now represents a patient, and can store
% variable number of visits
pet = mat2cell(pet(:, 2:end), counts);
labels = mat2cell(labels, counts);
% make each column an obervation rather than each row
pet = cellfun(@transpose, pet, 'UniformOutput', false);
labels = cellfun(@transpose, labels, 'UniformOutput', false);

% divide the data into training and testing sets
idx = randperm(numel(pet));
data.train = pet(idx(1:1000));
lab.train = labels(idx(1:1000));
data.test = pet(idx(1000:end));
lab.test = labels(idx(1000:end));

%% Generate ground truth data

dx = rowvec(sort(unique(stackedLabels)));
gt.pi = histc(stackedLabels, dx)/numel(stackedLabels)';
gt.A = zeros(numel(dx));
trans = cellfun(@countTransitions, labels, ...
                repmat({dx}, numel(labels), 1), 'UniformOutput', false);
gt.A = normalize(sum(cat(3, trans{:}), 3), 2);

K = 6;
Y = 3;

%% Train HMM model

[model, ll] = hmmFit(data.train, K, 'gauss', 'verbose', true, ...
                    'maxIter', 100, 'nRandomRestarts', 1);

path.train = getViterbiPath(data.train, model);
path.test = getViterbiPath(data.test, model);

dist.train = getStateDist(lab.train, path.train, Y, K);
dist.test = getStateDist(lab.test, path.test, Y, K);

[~, seq.train] = sort((dist.train(:, 1) + ones(size(dist.train(:, 1))))./   ...
                       (dist.train(:, 3) + ones(size(dist.train(:, 3)))), 'descend');
[~, seq.test] = sort((dist.test(:, 1) + ones(size(dist.test(:, 1))))./      ...
                       (dist.test(:, 3) + ones(size(dist.test(:, 3)))), 'descend');                   

if isequal(seq.train, seq.test)
    disp('Match in temporal sequence on training and testing set')
else
    disp('NO MATCH in temporal sequence on training and testing set')
end

% the natural ordering of the HMM states, heuristically determined
order = seq.train;
[model, path, dist] = getReordered(order, model, path, dist);

plotStateDist(dist.train, 'Distribution for training data');
plotStateDist(dist.test, 'Distribution for test set');

plotTrellis(data.train, path.train, lab.train, model, 'Trellis for training data');
plotTrellis(data.test, path.test, lab.test, model, 'Trellis for testing data');

end

function [model, path, dist] = getReordered(order, model, path, dist)
%% Reorder the model and results based on heuristic-based order of states

% order         -- The order of the states to use
% model         -- The final learned HMM model
% path          -- The Viterbi paths for both the training and test sets
% dist          -- The distribution over labels for each HMM states

%% Order the model

model.pi = model.pi(order);
A = zeros(size(model.A));
for i=1:size(model.A, 1)
    A(i, :) = model.A(order(i), order);
end

model.A = A;

%% Order the Viterbi paths

path.train = reorderPath(path.train);
path.test = reorderPath(path.test);
    
    function path = reorderPath(path)
        % flatten the path first
        stackedPath = cell2mat(path');
        % keep track of how big each path is
        counts = cellfun(@(seq)numel(seq), path);
        path = stackedPath;
        for k=1:numel(order)
            path(stackedPath == order(k)) = k;
        end
        path = mat2cell(colvec(path), counts);
        path = cellfun(@transpose, path, 'UniformOutput', false);
    end

%% Order the distribution matrices

dist.train = reorderDist(dist.train);
dist.test = reorderDist(dist.test);

    function dist = reorderDist(dist)
        newDist = zeros(size(dist));
        for k=1:numel(order)
            newDist(k, :) = dist(order(k), :);
        end
        dist = newDist;
    end

end

function [trellis, conv] = plotTrellis(data, path, labels, model, name)
%% Visualize the Viterbi Trellis

MCI = 2;
AD = 3;

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

function plotStateDist(h, name)
%% Plot a bar chart visualizing the distribution over labels for each state

%%
labNames = {'NL', 'MCI', 'AD'};

figure;

plt = bar(1:size(h, 1), h, 0.4, 'grouped');
grid on;
legend(plt, labNames);
title(name);
xlabel('HMM State number')
ylabel('Count')

end

function h = getStateDist(labels, path, Y, K)
%% Generate the distribution over ground-truth labels for every HMM state
% labels        -- A vector/cell containing the ground-truth sequence of labels
%                  for the observed data.
% path          -- A vector/cell containing the Viterbi sequence of states 
%                  for the corresponding labels.
%                  size(labels) = size(path)
% Y             -- The number of possible ground truth-states
% K             -- The number of states in the HMM

%%

if iscell(labels)
    labels = cell2mat(labels');
end

if iscell(path)
    path = cell2mat(path');
end

assert(isequal(size(labels), size(path)), ...
    'Dimensions of labels and path vector do not match!');

bins = 1:K;

% store counts of how the distribution over gt states is for every HMM
% state
h = zeros(Y, K);

for lab=1:Y
    h(lab, :) = hist(path(labels==lab), bins);
end

% every row represents a HMM state, every column is a gt label
h = h';

end

function path = getViterbiPath(data, model)
%% Viterbi sequence of paths for the observations of each patient
% data      -- Cell array, where each element contains the observations
%              seen for each patient
% model     -- The final model learnt by the HMM

%%
path = cell(size(data));

for i=1:numel(data)
    path{i} = hmmMap(model, data{i});
end

end

function trans = countTransitions(labels, dx)
%% Count the number of transitions of each type given the label sequence dx
% dx      -- The sequence of labels for a particular patient
% labels  -- The possible (label) states the patient can be in

%%
trans = zeros(numel(dx));
for i=1:numel(dx)
    % dx(i) is the source state
    % every element of dx will be the destination once
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



