function mu = kmeansPPInit(X, K)
%% Find K initial clusters from X using K-means++  
%
% X          ...   X(i, :) the ith data case
% K          ...   the number of clusters to fit

%% Perform seeding

disp('Seeding using K-means++');

[N, D] = size(X);
mu = ones(D, K);

% Randomly initialize first cluster
mu(:, 1) = X(randi(N), :)';

for i=2:K
    % find distance of every point from every mean chosen so far
    dist = sqDistance(X, mu(:, i-1)');
    % pick the smallest distance for each point, and build a pdf
    dist = cumsum(min(dist, 2));
    % normalize the pdf
    dist = dist/dist(end);
    % pick a point with probability given by the pdf
    mu(:, i) = X(find(rand < dist, 1, 'first'), :)';
end

end