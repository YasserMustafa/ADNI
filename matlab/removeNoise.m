function [data, labels] = removeNoise(data, labels)
%% Remove any patients with AD->MCI or AD->NL or NL->AD transitions

res = cellfun(@hasBadTransition, labels);

data = data(~res);
labels = labels(~res);

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