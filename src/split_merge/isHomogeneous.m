function tf = isHomogeneous(block, params)
% Predikat homogenitas P(R) untuk fase split
    C  = size(block, 3);
    px = reshape(block, [], C);          % setiap baris = satu piksel, kolom = kanal
    if size(px, 1) <= 1
        tf = true;
        return;
    end

    switch lower(params.SplitCriterion)
        case 'std'
            stat = std(px, 1, 1);                   % simpangan baku populasi per kanal
        case 'range'
            stat = max(px, [], 1) - min(px, [], 1); % rentang intensitas per kanal
        otherwise
            error('splitMerge:invalidParam', 'SplitCriterion tidak dikenal.');
    end
    tf = all(stat <= params.SplitThreshold);
end