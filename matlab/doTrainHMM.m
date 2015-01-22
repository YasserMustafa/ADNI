close all;

labelNames = {'NL', 'MCI', 'AD'};
[pet, convtime, labels, mmse, cdr] = getPetData(labelNames, 1);

% [dpet, data] = getDiffPhi(pet, {convtime, labels, mmse, cdr});
% [convtime, labels, mmse, cdr] = deal(data{:});

K = 6;

fprintf('Training PET HMM\n');
[~, ~, fdg.path, fdg.folds, fdg.idx] = hmm(pet, labels, ...
    'MMSE'              , mmse              , ...
    'CDR'               , cdr               , ...
    'K'                 , K                 , ...
    'labelNames'        , labelNames        , ...
    'ConvTime'          , convtime);

pause(1);
