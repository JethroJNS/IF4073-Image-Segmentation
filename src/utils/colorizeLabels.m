function rgb = colorizeLabels(labels)
% Beri warna semu (deterministik) pada setiap label region
    [H, W] = size(labels);
    N = max(labels(:));
    vals = mod((1:N)' * [0.6180339887 0.3819660113 0.2360679775], 1);
    cmap = 0.25 + 0.75 * vals;                  % hindari warna terlalu gelap
    rgb = zeros(H, W, 3);
    for c = 1:3
        rgb(:, :, c) = reshape(cmap(labels(:), c), H, W);
    end
    rgb = uint8(round(rgb * 255));
end