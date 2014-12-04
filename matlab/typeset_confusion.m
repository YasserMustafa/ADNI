function typeset_confusion(conf, stdev, K, Y)

counts = conf(end, :);
counts_std = stdev(end, :);

conf = conf(1:end-1, :);
stdev = stdev(1:end-1, :);

latIdx = find(any(conf, 2)); % non-zero rows
labIdx = find(any(conf, 1)); % non-zero columns

counts = counts(labIdx);
counts_std = counts_std(labIdx);

[latSrc, latDest] = ind2sub([K, K], latIdx);
[labSrc, labDest] = ind2sub([Y, Y], labIdx);

latLabels = cell(1, numel(latIdx));
labLabels = cell(1, numel(labIdx));

for i=1:numel(latIdx)
    src = latSrc(i);
    dest = latDest(i);
    latLabels{i} = horzcat([num2str(src), ' -> ', num2str(dest)]);
end

labelNames = {'NL', 'MCI', 'AD'};
for i=1:numel(labIdx)
    src = labSrc(i);
    dest = labDest(i);
    labLabels{i} = horzcat([labelNames{src}, ' -> ', labelNames{dest}]);
end

% get rid of rows with all zeros
conf = conf(latIdx, :);

% get rid of columns with all zeros
conf = conf(:, labIdx);

header = '\[ \begin{array}{ c ||';
format = repmat(' c |', 1, numel(labIdx));
header = horzcat([header, format(1:end-1), '}']);
disp(header);

text = ' &';
for col = 1:numel(labIdx)
    text = horzcat([text, ' ', labLabels{col}, ' &']);
end
text = horzcat([text(1:end-1), '\\']);
disp(text);
disp('\hline');
disp('\hline');

for row = 1:numel(latIdx)
    text = horzcat([latLabels{row}, ' &']);
    for col = 1:floor(numel(labIdx)/2)
        text = horzcat([text, ' ', num2str(conf(row, col), '%.3f'), ...
            ' \pm ', num2str(stdev(row, col), '%.3f'), ' &']);
    end
    text = horzcat([text(1:end-1), '\\']);
    disp(text);
    
    text = horzcat([latLabels{row}, ' &']);
    for col = floor(numel(labIdx)/2)+1:numel(labIdx)
        text = horzcat([text, ' ', num2str(conf(row, col), '%.3f'), ...
            ' \pm ', num2str(stdev(row, col), '%.3f'), ' &']);
    end
    text = horzcat([text(1:end-1), '\\']);
    disp(text);
end

text = 'Total Count &';
for col = 1:numel(labIdx)
    text = horzcat([text, ' ', num2str(counts(col), '%.3f'), ...
            ' \pm ', num2str(counts_std(col), '%.3f'), ' &']);
end
text = horzcat([text(1:end-1), '\\']);
disp(text);

disp('\end{array} \]');

end