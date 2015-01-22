function [dphi, data] = getDiffPhi(phi, data)
%% Take first derivative of feature matrix
% phi       -- Feature vector whose derivative should be taken
% data      -- Vector of cell arrays, each of which should be snipped 
%              by a visit to align with dPhi

%%
dphi = cellfun(@(seq)diff(seq, 1, 2), phi, 'UniformOutput', false);

% scale data by dividing by the max
stackedDphi = cell2mat(dphi')';
stackedDphi = stackedDphi/max(abs(stackedDphi(:)));
counts = cellfun(@(seq)size(seq, 2), dphi');

dphi = mat2cell(stackedDphi, counts);
dphi = cellfun(@transpose, dphi, 'UniformOutput', false);

for i=1:numel(data)
    data{i} = cellfun(@(seq)seq(:, 2:end), data{i}, ...
        'UniformOutput', false);
end

end