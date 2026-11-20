function F = fill_holes(B, keepLargestBg)
% Mengisi lubang pada citra biner
    if nargin < 2, keepLargestBg = false; end
    B = logical(B);

    [L, n] = label_components(~B, 4);
    if n == 0
        F = B;                                   % tidak ada background
        return;
    end

    keepLabels = unique([L(1, :), L(end, :), L(:, 1)', L(:, end)']);
    keepLabels = keepLabels(keepLabels > 0);

    if keepLargestBg && ~isempty(keepLabels)
        % Anggap background sebenarnya = komponen terbesar di antara yang menyentuh tepi
        area = accumarray(L(L > 0), 1, [n 1]);
        [~, i] = max(area(keepLabels));
        keepLabels = keepLabels(i);
    end

    holes = (~B) & ~ismember(L, keepLabels);
    F = B | holes;
end