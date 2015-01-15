function [conf, lat] = compareTerminalState(path, labels, K, Y)
%% Compare the terminal state of a Viterbi path with the corresponding clinical label sequence
%
% Input:
% path          -- path{i} contains the viterbi seq. for patient i
% labels        -- labels{i} contains the gt. labels for patient i
% K             -- number of states in the HMM
% Y             -- number of clinical labels
% Output:
% conf          -- The confusion matrix for the terminal state (K x Y)
% lat           -- the latency between the viterbi path and clinical labels

%%

% get the terminal state for both sequence types
term_state = cellfun(@(seq)seq(end), path);
term_label = cellfun(@(seq)seq(end), labels);

same = (term_state == term_label);
lat = findLatencies(path(same), labels(same));

% build confusion matrix
conf = confusionmat(term_label, term_state);
conf = conf(1:Y, 1:min(K, numel(unique(term_state))));

conf = normalize(conf, 2)';

end

function lat = findLatencies(path, labels)

lat = zeros(1, numel(path));

for i=1:numel(path)
    vit = path{i};
    lab = labels{i};
    assert (vit(end) == lab(end), 'Terminal states do not match!');
    vit_idx= find(vit == vit(end), 1, 'first');
    lab_idx = find(lab == lab(end), 1, 'first');
    lat(i) = vit_idx - lab_idx;
end

end