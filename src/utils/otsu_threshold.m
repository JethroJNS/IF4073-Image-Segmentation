function t = otsu_threshold(M)
% Ambang Otsu (memaksimalkan varians antar-kelas) untuk M di [0,1]
    M = min(max(M(:), 0), 1);
    idx = floor(M * 255) + 1;                 % bin 1..256
    h = accumarray(idx, 1, [256 1]);
    p = h / sum(h);
    w = cumsum(p);                            % bobot kelas 0
    mu = cumsum(p .* (0:255)');               % rerata kumulatif
    muT = mu(end);
    sigB = (muT * w - mu).^2 ./ (w .* (1 - w));
    sigB(~isfinite(sigB)) = 0;
    [~, k] = max(sigB);
    t = k / 255;                              % batas atas bin kelas 0
end