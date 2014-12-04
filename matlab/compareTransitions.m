function conf = compareTransitions(path, labels, K, Y)
%% Confusion-matrix to compare transitions in Viterbi state space to actual transitions for the gt labels
% 
% path          -- path{i} is the vector of the state sequence followed by 
%                  patient i
% labels        -- labels{i} is the vector of clinical states followed by 
%                  patient i
% K             -- number of states in the HMM (goes from 1 to K)
% Y             -- number of clinical labels (goes from 1 to Y)
% numel(labels{i}) = numel(path{i})

%%

% K*K possible transitions in the state space
% Y*Y possible transitions in the label space
conf = zeros(K*K, Y*Y);

assert(length(path) == length(labels), ...
    'Viterbi-path and gt-labels must have the same number of patients');

for i=1:numel(path)
    latent = path{i};
    label = labels{i};
    
    if numel(latent) > 1 && numel(label) > 1
        % we look at all the transitions in the two aligned sequences. Each
        % transition can be represented as srcIdx->destIdx, which lets us
        % treat (srcIdx, destIdx) as a cell in a [numStates numStates]
        % matrix. We can then use sub2ind to convert each transition tuple
        % into an integer index.

        latSrc = latent(1:end-1); % all the source states
        latDest = latent(2:end); % all the destination labels
        % all the transitions mapped to linear indices
        latIdx = sub2ind([K K], latSrc, latDest); 
    
        % same as above for gt labels
        labSrc = label(1:end-1);
        labDest = label(2:end);
        labIdx = sub2ind([Y Y], labSrc, labDest);

        % latIdx are the rows in the conf. matrix
        % labIdx are the cols in the conf. matrix
        confIdx = sub2ind(size(conf), latIdx, labIdx);
        idx = unique(confIdx);
        for k=1:numel(idx)
            conf(idx(k)) = conf(idx(k)) + sum(confIdx == idx(k));
        end
    end
end

counts = sum(conf, 1);
conf = [normalize(conf, 1); counts];

end