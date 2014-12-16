function labels = labelMCI(labels)
%% Label MCIs as converters/non-converters
%
% labels            -- labels{i} is the sequence of labels for patient i

%%

MCI = 2;
AD = 3;

MCI_c = 3;
MCI_nc = 2;
AD_new = 4;

for i=1:numel(labels)
    seq = labels{i};
    isMCI = any(seq == MCI);
    if isMCI
        mci_idx = find(seq == MCI, 1, 'first');
        ad_idx = find(seq == AD, 1, 'first');
        if ~isempty(ad_idx)
            assert( ad_idx > mci_idx, 'AD -> MCI transition observered!');
            seq(seq == AD) = AD_new;
            seq(seq == MCI) = MCI_c;
        else
            seq(seq == AD) = AD_new;
            seq(seq == MCI) = MCI_nc;
        end
    else
        seq(seq == AD) = AD_new;
    end
    labels{i} = seq;
end

end