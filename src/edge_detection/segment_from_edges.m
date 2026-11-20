function seg = segment_from_edges(E, params)
% Membentuk mask objek/region utama dari peta tepi biner
    if nargin < 2, params = default_edge_params(); end
    E = logical(E);
    N = numel(E);

    radii = params.closeRadii(:)';
    if isempty(radii), radii = 1; end            % jaga-jaga bila closeRadii kosong

    bestDist = inf; best = [];
    for r = radii
        SE = disk_element(r);
        closed = binary_dilate(E, SE);
        filled = fill_holes(closed, params.fillBorderObjects);
        filled = binary_erode(filled, SE);
        if params.openRadius > 0
            filled = binary_open(filled, disk_element(params.openRadius));
        end
        [labels, mask, nCand] = select_objects(filled, params);
        frac = nnz(mask) / N;

        % Jarak luas mask ke rentang wajar [minFrac, maxFrac] (0 = di dalam rentang)
        dist = max([params.minFrac - frac, frac - params.maxFrac, 0]);
        if frac == 0
            dist = dist + 1;                     % mask kosong selalu kalah dari yang tidak kosong
        end

        if dist < bestDist
            bestDist = dist;
            best.mask = mask; best.labels = labels; best.closed = closed;
            best.radius = r; best.areaFrac = frac; best.numCandidates = nCand;
        end
        if dist == 0, break; end
    end

    best.success = best.areaFrac > 0;
    best.valid   = best.success && (bestDist == 0);
    seg = best;
end

function [labels, mask, nCand] = select_objects(filled, params)
% Memilih komponen utama dengan skor berbasis luas
    [r, c] = size(filled);
    [L, n] = label_components(filled, 8);
    labels = zeros(r, c);
    mask = false(r, c);
    nCand = 0;
    if n == 0, return; end

    idx = find(L > 0);
    [ys, xs] = ind2sub([r c], idx);
    lab = L(idx);
    area = accumarray(lab, 1, [n 1]);
    cy = accumarray(lab, ys, [n 1]) ./ area;
    cx = accumarray(lab, xs, [n 1]) ./ area;

    % Bounding box tiap komponen
    minY = accumarray(lab, ys, [n 1], @min);  maxY = accumarray(lab, ys, [n 1], @max);
    minX = accumarray(lab, xs, [n 1], @min);  maxX = accumarray(lab, xs, [n 1], @max);
    extent = area ./ ((maxY - minY + 1) .* (maxX - minX + 1));

    % Jumlah sisi citra yang disentuh
    sides = (minY == 1) + (maxY == r) + (minX == 1) + (maxX == c);

    % Kedekatan ke pusat citra
    w  = params.centerWeight;
    d  = hypot((cy - (r + 1) / 2) / (r / 2), (cx - (c + 1) / 2) / (c / 2));
    fCenter = 1 - w + w * exp(-d.^2 / (2 * 0.7^2));

    fBorder = params.borderPenalty .^ sides;

    e = params.extentWeight;
    fBBox = 1 - e + e * extent;

    score = area .* fCenter .* fBorder .* fBBox;

    ok = area >= params.minAreaFrac * r * c;     % buang komponen terlalu kecil
    score(~ok) = 0;
    nCand = nnz(ok);

    [~, order] = sort(score, 'descend');
    order = order(score(order) > 0);
    order = order(1:min(params.numObjects, numel(order)));
    for k = 1:numel(order)
        labels(L == order(k)) = k;
    end
    mask = labels > 0;
end