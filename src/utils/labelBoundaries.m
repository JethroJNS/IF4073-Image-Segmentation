function bw = labelBoundaries(labels)
% Peta batas antar region
    [H, W] = size(labels);
    bw = false(H, W);

    diffRight = labels(:, 1:end-1) ~= labels(:, 2:end);
    bw(:, 1:end-1) = bw(:, 1:end-1) | diffRight;

    diffDown = labels(1:end-1, :) ~= labels(2:end, :);
    bw(1:end-1, :) = bw(1:end-1, :) | diffDown;
end