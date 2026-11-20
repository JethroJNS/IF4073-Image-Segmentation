function out = regionMeanImage(X, labels)
% Warnai setiap region dengan warna rata-rata pikselnya
    [H, W, C] = size(X);
    N = max(labels(:));
    cnt = accumarray(labels(:), 1, [N 1]);
    out = zeros(H, W, C);
    for c = 1:C
        ch = X(:, :, c);
        meanVal = accumarray(labels(:), ch(:), [N 1]) ./ cnt;   % rata-rata per region
        out(:, :, c) = reshape(meanVal(labels), H, W);
    end
    out = uint8(round(out));
end