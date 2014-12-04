function [val, counts] = findLatencies(path, labels)

% only want to consider sequences with zero or 1 transitions
idx = cellfun(@(seq)sum(diff(seq)~=0), labels);
longest = max(cellfun(@(seq)numel(seq), labels));
labels = labels(idx <= 1);
path = path(idx <= 1);

val = -longest:longest+1;
counts = zeros(size(val));

for i=1:numel(path)
    lab = labels{i};
    vit = path{i};

    labIdx = find(lab==lab(end), 1, 'first');
    vitIdx = find(vit==vit(end), 1, 'first');
    
    if ~isempty(vitIdx)
        lat = vitIdx - labIdx;
        idx = val==lat;
        counts(idx) = counts(idx) + 1;
    else
        counts(end) = counts(end) + 1;
    end
end

end