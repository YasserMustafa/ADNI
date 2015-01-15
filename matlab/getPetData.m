function [pet, clinical, labels, mmse, cdr] = getPetData(labelNames)
%% Read and clean data

%%

% read in and clean the FDG-PET scan data, and condition it to be used by
% the pmtk3 package

data_pet = '/phobos/alzheimers/adni/pet_mmse_cdr.csv';
%data_pet = '/Users/Dev/Documents/ADNI/data/pet_mmse_cdr.csv';

% must use this because there is some non-numeric data in table (e.g.
% VISCODE2)
pet = readtable(data_pet);

% grab the DX for each patient
labels = pet.DX;
labels = getLabels(labels, labelNames);

mmse = pet.MMSCORE; % MMSE scores
cdr = pet.CDGLOBAL; % CDR ratings
idx = mmse~=-1 & ~isnan(mmse) & cdr~=-1 & ~isnan(cdr);

% ignore VISCODE2 for now, which is column #2
% the high dimensionality of the data is causing singular problems with the
% covariance matrices. Just work with means for now.
% 4=MEAN, 5=MEDIAN, 6=MODE, 7=MIN, 8=MAX, 9=STDEV
pet = table2array(pet(:, [1, 6:6:end])); 
pet = pet(idx, :);
labels = labels(idx);
mmse = mmse(idx);
cdr = cdr(idx);
clinical = [mmse cdr];

% When using pet_average_flat, we want all features
%pet = table2array(pet(:, [1, 4:end]));

% divide up the data by RIDs
% RID = data(:, 1)
[~, order] = sort(pet(:, 1)); % first sort based on RID
pet = pet(order, :); % make sure the RIDs appear in ascending order
% generate counts for each RID
counts = histc(pet(:, 1), unique(pet(:, 1))); 

% get rid of the RID
pet = pet(:, 2:end);

% counts will have the number of rows that belong to each RID
% generate a cell array now, using these counts to divide up the matrix
% each cell of this cell array now represents a patient, and can store
% variable number of visits
pet = mat2cell(pet, counts);
labels = mat2cell(labels, counts);
mmse = mat2cell(mmse, counts);
cdr = mat2cell(cdr, counts);
clinical = mat2cell(clinical, counts);
% make each column an obervation rather than each row
pet = cellfun(@transpose, pet, 'UniformOutput', false);
labels = cellfun(@transpose, labels, 'UniformOutput', false);
mmse = cellfun(@transpose, mmse, 'UniformOutput', false);
cdr = cellfun(@transpose, cdr, 'UniformOutput', false);
clinical = cellfun(@transpose, clinical, 'UniformOutput', false);

minVists = 1;
[data, labels] = removeNoise({pet, clinical, mmse, cdr}, labels, minVists);
[pet, clinical, mmse, cdr] = deal(data{:});

end
%% Helper functions

function labels = getLabels(labels, labelNames)
%% Convert the string labels in the data to numerical labels

%%

nl = cellfun(@strcmp, labels, repmat({'NL'}, size(labels, 1), 1));
ad = cellfun(@strcmp, labels, repmat({'AD'}, size(labels, 1), 1));
mci_nc = cellfun(@strcmp, labels, repmat({'MCI-NC'}, size(labels, 1), 1));
mci_c = cellfun(@strcmp, labels, repmat({'MCI-C'}, size(labels, 1), 1));
mci = cellfun(@strcmp, labels, repmat({'MCI'}, size(labels, 1), 1));

labels = zeros(size(labels));

labels(nl) = 1;

if isequal(labelNames, {'NL', 'MCI-NC', 'MCI-C', 'AD'})
    labels(mci_nc) = 2;
    labels(mci_c) = 3;
    labels(ad) = 4;
elseif isequal(labelNames, {'NL', 'MCI', 'AD'})
    labels(mci_nc) = 2;
    labels(mci_c) = 2;
    labels(mci) = 2;
    labels(ad) = 3;
end

end