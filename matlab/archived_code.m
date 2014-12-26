function archived_code()

end

function trellis = getTrellis(path, K, t_max)
%% Visualize the Viterbi Trellis

% store the number of transitions from every possible source to every
% possible destination. The row represents the source, and the column the 
% destination state. The third dimension 't' represents the transitions seen
% at time = t.
trellis = zeros(K, K, t_max-1);

for i=1:numel(path)
    for t=1:numel(path{i})-1
        r = path{i}(t);
        c = path{i}(t+1);
        trellis(r, c, t) = trellis(r, c, t) + 1;
    end
end

end

function plotTrellis(trellis, name)

t_max = size(trellis, 3);
K = size(trellis, 1);

figure;
hold on;
xlim([0, t_max+1])
ylim([0, K+1])
xlabel('Time')
ylabel('HMM State')

for t=1:size(trellis, 3)
    [r, c] = find(trellis(:, :, t));
    for i=1:numel(r)
        trans = trellis(:, :, t);
        normalizer = sum(trans(:));
        plot([t, t+1], [r(i), c(i)], ...
            'Marker', '.', 'MarkerSize', 10, ...
            'LineStyle', '-', 'LineWidth', 30*trans(r(i), c(i))/normalizer);
        alpha(0.2);
    end
end
hold off;

title(name);

end

function typeset_trans(A)
%% Print a LaTeX version of a transition matrix learned by the HMM

%%
A = cat(3, A{:});
val = mean(A, 3);
stdev = std(A, 0, 3);
fprintf('Transition matrix\n')

for row = 1:size(val, 1)
    text = '';
    for col = 1:size(val, 2)
        text = horzcat([text, ' ', num2str(val(row, col), '%.3f'), ...
            ' \pm ', num2str(stdev(row, col), '%.3f'), ' &']);
    end
    text = horzcat([text(1:end-1), '\\']);
    disp(text);
end

end