% read in and clean the FDG-PET scan data, and condition it to be used by
% the pmtk3 package

data_pet = '/phobos/alzheimers/adni/pet_flat.csv';

% must use this because there is some non-numeric data in table (e.g.
% VISCODE2)
data = readtable(data_pet);

% ignore VISCODE2 for now, which is column #2
data = table2array(data(:, [1, 3:end]));

% divide up the data by RIDs
% RID = data(:, 1)
[~, order] = sort(data(:, 1)); % first sort based on RID
data = data(order, :); % make sure the RIDs appear in ascending order
% generate counts for each RID
counts = histc(data(:, 1), unique(data(:, 1))); 
% counts will have the number of rows that belong to each RID
% generate a cell array now, using these counts to divide up the matrix
data = mat2cell(data(:, 2:end), counts);
% make each column an obervation rather than each row
data = cellfun(@transpose, data, 'UniformOutput', false);

[model, ll] = hmmFit(data, 5, 'gauss');