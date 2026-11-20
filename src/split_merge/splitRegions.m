function [labels, numRegions, rects] = splitRegions(img, params)
% Pembelahan quadtree rekursif
    [H, W, ~] = size(img);
    mb  = params.MinBlockSize;
    buf = zeros(1024, 4);                % buffer daun [r1 r2 c1 c2]
    cnt = 0;                             % jumlah daun yang sudah tercatat

    splitBlock(1, H, 1, W);

    rects      = buf(1:cnt, :);
    numRegions = cnt;

    labels = zeros(H, W);
    for k = 1:numRegions
        labels(rects(k,1):rects(k,2), rects(k,3):rects(k,4)) = k;
    end

    function splitBlock(r1, r2, c1, c2)
        h = r2 - r1 + 1;
        w = c2 - c1 + 1;

        canSplitRows = h > mb;
        canSplitCols = w > mb;

        % Kasus dasar: tidak bisa dibelah lagi, atau sudah homogen
        if (~canSplitRows && ~canSplitCols) || ...
                isHomogeneous(img(r1:r2, c1:c2, :), params)
            addLeaf(r1, r2, c1, c2);
            return;
        end

        % DIVIDE
        if canSplitRows
            rm = floor((r1 + r2) / 2);
            rowSpans = [r1 rm; rm+1 r2];
        else
            rowSpans = [r1 r2];
        end
        if canSplitCols
            cm = floor((c1 + c2) / 2);
            colSpans = [c1 cm; cm+1 c2];
        else
            colSpans = [c1 c2];
        end

        % CONQUER
        for i = 1:size(rowSpans, 1)
            for j = 1:size(colSpans, 1)
                splitBlock(rowSpans(i,1), rowSpans(i,2), ...
                           colSpans(j,1), colSpans(j,2));
            end
        end
    end

    function addLeaf(r1, r2, c1, c2)
        cnt = cnt + 1;
        if cnt > size(buf, 1)
            buf(2 * size(buf, 1), 4) = 0;
        end
        buf(cnt, :) = [r1 r2 c1 c2];
    end
end