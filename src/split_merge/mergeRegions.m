function [labels, numRegions] = mergeRegions(img, labels, params)
% Gabungkan region bertetangga yang similar
    [H, W, C] = size(img);
    px  = reshape(img, [], C);
    lab = labels(:);
    N   = max(lab);

    % Ruang warna untuk perbandingan rata-rata (Lab hanya untuk citra 3 kanal)
    if C == 3 && strcmp(params.ColorSpace, 'lab')
        toSpace = @rgbToLab;
    else
        toSpace = @(x) x;
    end

    % Statistik tiap region: jumlah piksel, jumlah intensitas, jumlah kuadrat
    n  = accumarray(lab, 1, [N 1]);
    s  = zeros(N, C);
    ss = zeros(N, C);
    for c = 1:C
        s(:, c)  = accumarray(lab, px(:, c),      [N 1]);
        ss(:, c) = accumarray(lab, px(:, c) .^ 2, [N 1]);
    end

    parent = (1:N)';                         % union-find
    pairs  = regionAdjacency(labels);
    T      = params.MergeThreshold;
    maxStd = params.MergeStdMax;

    % Merge berdasarkan similarity
    while true
        pairs = rootPairs(pairs, parent);
        if isempty(pairs), break; end

        d    = meanDistance(toSpace(s(pairs(:,1), :) ./ n(pairs(:,1))), ...
                            toSpace(s(pairs(:,2), :) ./ n(pairs(:,2))));
        cand = find(d <= T);
        if isempty(cand), break; end
        [~, ord] = sort(d(cand));            % paling mirip lebih dulu
        cand = cand(ord);

        merged = false;
        for k = cand'
            a = findRoot(parent, pairs(k, 1));
            b = findRoot(parent, pairs(k, 2));
            if a == b, continue; end

            % hitung ulang jarak dengan statistik terkini
            if meanDistance(toSpace(s(a,:) / n(a)), toSpace(s(b,:) / n(b))) > T
                continue;
            end

            nn   = n(a) + n(b);
            ssum = s(a,:) + s(b,:);
            ssq  = ss(a,:) + ss(b,:);
            if isfinite(maxStd)
                v = ssq / nn - (ssum / nn) .^ 2;
                if sqrt(max(max(v, 0))) > maxStd, continue; end
            end

            if n(a) < n(b), t = a; a = b; b = t; end   % union by size
            parent(b) = a;
            n(a) = nn;  s(a,:) = ssum;  ss(a,:) = ssq;
            merged = true;
        end
        if ~merged, break; end
    end

    % Hapus region yang terlalu kecil
    minSize = params.MinRegionSize;
    if isinf(params.SmallRegionDistFactor)
        maxSmallDist = Inf;                  % hindari Inf * 0 = NaN
    else
        maxSmallDist = params.SmallRegionDistFactor * T;
    end

    while minSize > 1
        pairs  = rootPairs(pairs, parent);
        isRoot = (parent == (1:N)');
        small  = isRoot & (n < minSize);
        if ~any(small) || isempty(pairs), break; end

        d  = meanDistance(toSpace(s(pairs(:,1), :) ./ n(pairs(:,1))), ...
                          toSpace(s(pairs(:,2), :) ./ n(pairs(:,2))));
        dp = [pairs(:,1) pairs(:,2) d; pairs(:,2) pairs(:,1) d];
        dp = dp(small(dp(:,1)) & dp(:,3) <= maxSmallDist, :);
        if isempty(dp), break; end
        dp = sortrows(dp, [1 3]);
        [~, first] = unique(dp(:,1), 'first');
        best = dp(first, 1:2);
        [~, ord] = sort(n(best(:,1)));
        best = best(ord, :);

        merged = false;
        for k = 1:size(best, 1)
            a = findRoot(parent, best(k, 1));
            b = findRoot(parent, best(k, 2));
            if a == b, continue; end
            if n(a) >= minSize && n(b) >= minSize, continue; end   % sudah tidak kecil
            if isfinite(maxSmallDist) && ...
                    meanDistance(toSpace(s(a,:) / n(a)), toSpace(s(b,:) / n(b))) > maxSmallDist
                continue;
            end
            if n(a) < n(b), t = a; a = b; b = t; end
            parent(b) = a;
            n(a) = n(a) + n(b);  s(a,:) = s(a,:) + s(b,:);  ss(a,:) = ss(a,:) + ss(b,:);
            merged = true;
        end
        if ~merged, break; end
    end

    % Pelabelan ulang berurutan 1..M
    roots = fullRoots(parent);
    lab2  = roots(labels);
    [u, ~, ic] = unique(lab2(:));
    labels     = reshape(ic, H, W);
    numRegions = numel(u);
end

function r = findRoot(parent, x)
% Cari akar union-find
    while parent(x) ~= x
        x = parent(x);
    end
    r = x;
end

function p = fullRoots(parent)
% Akar setiap elemen sekaligus
    p = parent;
    while any(p(p) ~= p)
        p = p(p);
    end
end

function pairs = rootPairs(pairs, parent)
% Petakan pasangan tetangga ke akar region saat ini, buang duplikat/diri sendiri
    if isempty(pairs), pairs = zeros(0, 2); return; end
    roots = fullRoots(parent);
    a = roots(pairs(:,1));
    b = roots(pairs(:,2));
    keep = a ~= b;
    pairs = unique(sort([a(keep) b(keep)], 2), 'rows');
    if isempty(pairs), pairs = zeros(0, 2); end
end

function pairs = regionAdjacency(labels)
% Pasangan label bertetangga, a < b, tanpa duplikat
    a = labels(:, 1:end-1);  b = labels(:, 2:end);   m = a ~= b;
    p1 = [a(m) b(m)];
    a = labels(1:end-1, :);  b = labels(2:end, :);   m = a ~= b;
    p2 = [a(m) b(m)];
    pairs = [p1; p2];
    if isempty(pairs), pairs = zeros(0, 2); return; end
    pairs = unique(sort(pairs, 2), 'rows');
end

function d = meanDistance(x, y)
% Jarak RMS antar rata-rata warna
    d = sqrt(mean((x - y) .^ 2, 2));
end

function lab = rgbToLab(rgb)
% Konversi sRGB ke CIE L*a*b*
    c   = rgb / 255;
    lin = c / 12.92;
    hi  = c > 0.04045;
    lin(hi) = ((c(hi) + 0.055) / 1.055) .^ 2.4;     % linearisasi gamma sRGB

    M   = [0.4124564 0.3575761 0.1804375; ...
           0.2126729 0.7151522 0.0721750; ...
           0.0193339 0.1191920 0.9503041];
    xyz = (lin * M') ./ [0.95047 1.0 1.08883];      % dinormalisasi terhadap titik putih

    f  = xyz;
    th = 216 / 24389;
    big = xyz > th;
    f(big)  = xyz(big) .^ (1/3);
    f(~big) = (24389/27 * xyz(~big) + 16) / 116;

    lab = [116 * f(:,2) - 16, ...
           500 * (f(:,1) - f(:,2)), ...
           200 * (f(:,2) - f(:,3))];
end