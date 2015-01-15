function writeViterbiProb(prob, labels, mmse, cdr, idx, name)

data = [];

for f = 1:numel(prob)
    stackedProb = cell2mat(cellfun(@(seq)seq(:, end), prob{f}', ...
        'UniformOutput', false));
    stackedLabel = cell2mat(cellfun(@(seq)seq(:, end), labels(idx{f})', ...
        'UniformOutput', false));
    stackedMMSE = cell2mat(cellfun(@(seq)seq(:, end), mmse(idx{f})', ...
        'UniformOutput', false));
    stackedCDR = cell2mat(cellfun(@(seq)seq(:, end), cdr(idx{f})', ...
        'UniformOutput', false));
    d = [stackedProb; stackedLabel; stackedMMSE; stackedCDR];
    data = [data; d'];
end

csvwrite(name, data);

end