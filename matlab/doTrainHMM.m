close all;

%labelNames = {'NL', 'MCI', 'AD'};
labelNames = {'NL', 'MCI-NC', 'MCI-C', 'AD'};
[pet, clinical, labels, mmse, cdr] = getPetData(labelNames);

K = 6;

fprintf('Training PET HMM\n');
[~, ~, pet_path, pet_folds, idx] = hmm(pet, labels, mmse, cdr, K, ...
    labelNames, zeros(1, numel(pet)));

% K=3;

% fprintf('Training MMSE HMM\n');
% [~, ~, mmse_path, mmse_folds, ~] = hmm(mmse, labels, mmse, cdr, K, ...
%     labelNames, idx);