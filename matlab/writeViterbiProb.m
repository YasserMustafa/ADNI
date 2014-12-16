function writeViterbiProb(prob, labels, idx, name)

data = [];

for f = 1:numel(prob)
    d = [cell2mat(prob{f}'); cell2mat(labels(idx{f})')];
    data = [data; d'];
end

csvwrite(name, data);

end