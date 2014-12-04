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
% 4=MEAN, 5=MEDIAN, 6=MODE, 7=MIN, 8=MAX, 9=STDEV
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

[pet, labels] = removeNoise(pet, labels);

%% Generate ground truth data

dx = rowvec(sort(unique(stackedLabels)));
gt.pi = histc(stackedLabels, dx)/numel(stackedLabels)';
gt.A = normalize(countTransitions(labels, numel(dx)), 2);

Y = 3;
K = 9;

numFolds = 3; % number of CV folds
num = floor(length(labels)/numFolds); % number of testing examples in each fold
idx = randperm(numel(pet));

% first row is training, second row is testing
% each column belongs to a particular fold
loglik= cell(2, numFolds);
path = cell(2, numFolds);
dist = cell(2, numFolds);
A = cell(1, numFolds);
pi = cell(1, numFolds);
conf = cell(2, numFolds);

%% Perform CV 
for fold=1:numFolds
    %% divide data into training/testing
    testIdx = num*(fold-1) + 1:min(num*fold, length(labels));
    trainIdx = setxor(1:numel(labels), testIdx);
    fprintf('Fold %d: %d training, %d testing\n', fold, numel(trainIdx), numel(testIdx));
    data.train = pet(idx(trainIdx));
    data.test = pet(idx(testIdx));
    lab.train = labels(idx(trainIdx));
    lab.test = labels(idx(testIdx));

    %% Train HMM model
    [model, ll] = hmmFit(data.train, K, 'gauss', 'verbose', true, ...
        'maxIter', 100, 'nRandomRestarts', 1, ...
        'transPrior', zeros(K));
    
    loglik{1, fold} = getLogLik(model, data.train);
    loglik{2, fold} = getLogLik(model, data.test);
    
    path{1, fold} = getViterbiPath(data.train, model);
    path{2, fold} = getViterbiPath(data.test, model);
    
    dist{1, fold} = getStateDist(lab.train, path{1, fold}, Y, K);
    dist{2, fold} = getStateDist(lab.test, path{2, fold}, Y, K);

    % the natural ordering of the HMM states, heuristically determined
    [~, seq] = sort((dist{1, fold}(:, 1) + ones(size(dist{1, fold}(:, 1))))./   ...
        (dist{1, fold}(:, 3) + ones(size(dist{1, fold}(:, 3)))), 'descend');
    
    [model, path(:, fold), dist(:, fold)] = getReordered(seq, model, ...
        path(:, fold), dist(:, fold));
    
    A{fold} = model.A;
    pi{fold} = model.pi;
    
    conf{1, fold} = compareTransitions(path{1, fold}, lab.train, K, Y);
    conf{2, fold} = compareTransitions(path{2, fold}, lab.test, K, Y);
end

close all;
train_dist = cat(3, dist{1, :});
test_dist = cat(3, dist{2, :});
plotStateDist(mean(train_dist, 3), std(train_dist, 0, 3), 'Distribution for training set');
plotStateDist(mean(test_dist, 3), std(test_dist, 0, 3), 'Distribution for test set');

train_conf = cat(3, conf{1, :});
test_conf = cat(3, conf{2, :});

fprintf('Training confusion\n')
typeset_confusion(mean(train_conf, 3), std(train_conf, 0, 3), K, Y);
fprintf('Testing confusion\n')
typeset_confusion(mean(test_conf, 3), std(test_conf, 0, 3), K, Y);

[val, counts] = findLatencies(getViterbiPath(pet, model), labels);
figure;
hold on;
bar(val, counts);
title('Latency in reaching terminal state');
xlabel('Latency');
ylabel('Count');
hold off;

fprintf('Transition matrix\n')
A = cat(3, A{:});
A_mean = mean(A, 3);
A_std = std(A, 0, 3);
typeset_trans(A_mean, A_std);

pi = cat(3, pi{:});
pi_mean = mean(pi, 3);
pi_std = std(pi, 0, 3);

% plotTrellis(path.train, K, 'Trellis for training data');
% plotTrellis(path.test, K, 'Trellis for testing data');

% plotTrellis(labels, K, 'Trellis for ground-truth data');

end

function typeset_trans(val, stdev)

for row = 1:size(val, 1)
    text = '';
    for col = 1:size(val, 2)
        text = horzcat([text, ' ', num2str(val(row, col), '%.3f'), ...
            ' \pm ', num2str(stdev(row, col), '%.3f'), ' &']);
    end
    text = horzcat([text(1:end-1), '\\']);
    disp(text);
end

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
mu = zeros(size(model.emission.mu));
Sigma = zeros(size(model.emission.Sigma));

for i=1:size(model.A, 1)
    A(i, :) = model.A(order(i), order);
    mu(:, i) = model.emission.mu(:, order(i));
    Sigma(:, :, i) = model.emission.Sigma(:, :, order(i));
end

model.A = A;
model.emission.mu = mu;
model.emission.Sigma = Sigma;

%% Order the Viterbi paths

for i=1:numel(path)
    path{i}= reorderPath(path{i});
end

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

for i=1:numel(dist)
    dist{i}= reorderDist(dist{i});
end

    function dist = reorderDist(dist)
        newDist = zeros(size(dist));
        for k=1:numel(order)
            newDist(k, :) = dist(order(k), :);
        end
        dist = newDist;
    end

end

function trellis = plotTrellis(path, K, name)
%% Visualize the Viterbi Trellis

t_max = max(cellfun(@(x)length(x), path)); % the longest obs. seq. over all patients

% store the number of transitions from every possible source to every
% possible destination. The row represents the source, and the column the 
% destination state. The matrix in cell {t} represents the transitions seen
% at time = t.
trellis = repmat({zeros(K)}, t_max-1, 1);

for i=1:numel(path)
    for t=1:numel(path{i})-1
        r = path{i}(t);
        c = path{i}(t+1);
        trellis{t}(r, c) = trellis{t}(r, c) + 1;
    end
end

figure;
hold on;
xlim([0, t_max+1])
ylim([0, K+1])
xlabel('Time')
ylabel('HMM State')

for t=1:numel(trellis)
    [r, c] = find(trellis{t});
    for i=1:numel(r)
        normalizer = sum(trellis{t}(:));
        plot([t, t+1], [r(i), c(i)], ...
            'Marker', '.', 'MarkerSize', 10, ...
            'LineStyle', '-', 'LineWidth', 150*trellis{t}(r(i), c(i))/normalizer);
        alpha(0.2);
    end
end
hold off;

title(name);

end

function plotStateDist(dist, stdev, name)
%% Plot a bar chart visualizing the distribution over labels for each state

%%
labNames = {'NL', 'MCI', 'AD'};

errorbar_groups(dist', stdev', 'bar_width', 0.75);
%plt = bar(1:size(h, 1), h, 0.4, 'grouped');
grid on;
legend(labNames);
title(name);
xlabel('HMM State number')
ylabel('Fraction of cases')

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

h = normalize(h, 2);
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


