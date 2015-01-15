function [model, ll, path, folds, idx] = hmm(phi, labels, mmse, cdr, K, labelNames, idx)
%% Train a HMM model on the PET data, and visualize the results

%% Read and clean data

% read in and clean the FDG-PET scan data, and condition it to be used by
% the pmtk3 package

%% Generate ground truth data

stackedLabels = cell2mat(labels')';
dx = rowvec(sort(unique(stackedLabels)));
gt.pi = histc(stackedLabels, dx)/numel(stackedLabels)';
gt.A = normalize(countTransitions(labels, numel(dx)), 2);

Y = numel(labelNames);
t_max = max(cellfun(@(seq)(numel(seq)), labels));

% number of CV folds
numFolds = 3; 
% number of testing examples in each fold
num = floor(length(labels)/numFolds);

% random permutation of all data instances
new_idx = randperm(numel(phi)); 
if isequal(idx, zeros(size(new_idx)))
    idx = new_idx;
end

% first row is training, second row is testing
% each column belongs to a particular fold

% Loglik of the data
loglik = cell(2, numFolds); 
% The viterbi path given the observations
path = cell(2, numFolds); 
% The prob. seq. of each path derived from the viterbi
prob = cell(2, numFolds); 
% the distribution of labels over each HMM state
dist = cell(2, numFolds); 
% The transition matrix learned by the HMM
A = cell(1, numFolds); 
% The initial state dist. learned by the HMM
pi = cell(1, numFolds); 
% Confusion matrix to map transitions in clinical labels 
% to transitions in the HMM states
conf = cell(2, numFolds); 
% Confusion matrix for terminal state in HMM vs clinical labels
term = cell(2, numFolds); 
% Indices of training/testing set for each fold
folds = cell(2, numFolds); 

%% Perform CV 
for fold=1:numFolds
    %% divide data into training/testing
    testIdx = num*(fold-1) + 1:min(num*fold, length(labels));
    trainIdx = setxor(1:numel(labels), testIdx);
    fprintf('Fold %d: %d training, %d testing\n', ...
        fold, numel(trainIdx), numel(testIdx));
    data.train = phi(idx(trainIdx));
    data.test = phi(idx(testIdx));
    lab.train = labels(idx(trainIdx));
    lab.test = labels(idx(testIdx));
    folds{1, fold} = idx(trainIdx);
    folds{2, fold} = idx(testIdx);

    %% Train HMM model
    [model, ll] = hmmFit(data.train, K, 'gauss', 'verbose', true, ...
        'maxIter', 100, 'nRandomRestarts', 1, ...
        'transPrior', zeros(K));
    
    loglik{1, fold} = getLogLik(model, data.train);
    loglik{2, fold} = getLogLik(model, data.test);
    
    [path{1, fold}, prob{1, fold}] = getViterbiPath(data.train, model);
    [path{2, fold}, prob{2, fold}] = getViterbiPath(data.test, model);
    
    dist{1, fold} = getStateDist(lab.train, path{1, fold}, Y, K);
    dist{2, fold} = getStateDist(lab.test, path{2, fold}, Y, K);

    % the natural ordering of the HMM states, heuristically determined
    [~, seq] = sort((dist{1, fold}(:, 1) + ...
        ones(size(dist{1, fold}(:, 1))))./ (dist{1, fold}(:, 3) + ...
        ones(size(dist{1, fold}(:, 3)))), 'descend');
    
    [model, path(:, fold), prob(:, fold), dist(:, fold)] = ...
        getReordered(seq, model, ...
        path(:, fold), prob(:, fold), dist(:, fold));
    

    A{fold} = model.A;
    pi{fold} = model.pi;
    
    conf{1, fold} = compareTransitions(path{1, fold}, lab.train, K, Y);
    conf{2, fold} = compareTransitions(path{2, fold}, lab.test, K, Y);
    
    term{1, fold} = compareTerminalState(path{1, fold}, lab.train, K, Y);
    term{2, fold} = compareTerminalState(path{2, fold}, lab.test, K, Y);
end

%% state distribution across clinical labels

train_dist = cat(3, dist{1, :});
test_dist = cat(3, dist{2, :});
plotStateDist(mean(train_dist, 3), std(train_dist, 0, 3), ...
labelNames, 'Training Dist.(aggregated over time)');
plotStateDist(mean(test_dist, 3), std(test_dist, 0, 3), ...
labelNames, 'Testing Dist.(aggregated over time)');

%% terminal state distribution across clinical labels
train_term = cat(3, term{1, :});
test_term = cat(3, term{2, :});
plotStateDist(mean(train_term, 3), std(train_term, 0, 3), ...
labelNames, 'Training Dist.(Terminal State)');
plotStateDist(mean(test_term, 3), std(test_term, 0, 3), ...
labelNames, 'Testing Dist.(Terminal State)');

%% MMSE dis. across HMM states

showStateClinicalDist(mmse, folds(1, :), path(1, :), K, ...
    'MMSE Dist. (Training)');
showStateClinicalDist(mmse, folds(2, :), path(2, :), K, ...
    'MMSE Dist. (Test)');

%% CDR dis. across HMM states

showStateClinicalDist(cdr, folds(1, :), path(1, :), K, ...
    'CDR Dist. (Training)');
showStateClinicalDist(cdr, folds(2, :), path(2, :), K, ...
    'CDR Dist. (Test)');

%% write Viterbi prob. to csv

writeViterbiProb(prob(1, :), labels, mmse, cdr, folds(1, :), ...
    'viterbi_prob_train.csv');
writeViterbiProb(prob(2, :), labels, mmse, cdr, folds(2, :), ...
    'viterbi_prob_test.csv');

%% Confusion matrices

% train_conf = cat(3, conf{1, :});
% test_conf = cat(3, conf{2, :});

% fprintf('Training confusion\n')
% typeset_confusion(mean(train_conf, 3), std(train_conf, 0, 3), K, Y);
% fprintf('Testing confusion\n')
% typeset_confusion(mean(test_conf, 3), std(test_conf, 0, 3), K, Y);

end

%% Helper functions

function [model, path, prob, dist] = getReordered(order, model, path,...
    prob, dist)
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

%% Order the Viterbi paths and probability vectors

for i=1:numel(path)
    [path{i}, prob{i}] = reorderPath(path{i}, prob{i});
end

    function [path, prob] = reorderPath(path, prob)
        % flatten the path first
        stackedPath = cell2mat(path');
        stackedProb = cell2mat(prob');
        % keep track of how big each path is
        counts = cellfun(@(seq)numel(seq), path);
        path = stackedPath;
        prob = stackedProb;
        for k=1:numel(order)
            path(stackedPath == order(k)) = k;
            prob(k, :) = stackedProb(order(k), :);
        end
        path = mat2cell(colvec(path), counts);
        prob = mat2cell(prob, size(prob, 1), counts);
        path = cellfun(@transpose, path, 'UniformOutput', false);
        prob = prob';
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

function plotStateDist(dist, stdev, labNames, name)
%% Plot a bar chart visualizing the distribution over labels for each state

%%

figure;
bar(1:size(dist, 1), dist, 0.4, 'grouped');
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

function [path, prob] = getViterbiPath(data, model)
%% Viterbi sequence of paths for the observations of each patient
% data      -- Cell array, where each element contains the observations
%              seen for each patient
% model     -- The final model learnt by the HMM
% prob      -- The probability of being in each state at a given time

%%
path = cell(size(data));
prob = cell(size(path));

for i=1:numel(data)
    [path{i}, prob{i}] = hmmMap(model, data{i});
end

end


