function [data, labels] = removeNoise(data, labels, minVisits)
%% Remove any patients with AD->MCI or AD->NL or NL->AD transitions
%
% data          -- data{i} are the observations for patient 'i'
% labels        -- labels{i} is the seq. of clinical labels for patient 'i'
% minVisits     -- keep only patients with at least 'minVisits' visits

bad = cellfun(@hasBadTransition, labels);
visits = cellfun(@(seq)numel(seq) >= minVisits, labels);

for i=1:numel(data)
    data{i} = data{i}(~bad & visits);
end
labels = labels(~bad & visits);

end

function res = hasBadTransition(trajectory)
%% Does the patient have one of the bad transitions mentioned above?

%% define codes

AD = 3;
MCI = 2;
NL = 1;

%% Check for AD->MCI

ad = find(trajectory == AD);
nl = find(trajectory == NL);

ad = ad(ad < numel(trajectory)); 
nl = nl(nl < numel(trajectory)); 

res = any(trajectory(ad+1) == MCI) || any(trajectory(ad+1) == NL) || ...
    any(trajectory(nl+1) == AD);

end