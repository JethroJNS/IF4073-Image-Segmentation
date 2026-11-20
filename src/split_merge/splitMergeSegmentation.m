function [labels, segmented, info] = splitMergeSegmentation(img, varargin)
% Segmentasi citra klasik dengan split and merge
    params = splitMergeParams(varargin{:});

    X = imageToDouble255(img, params.FloatRange);
    if strcmp(params.ColorMode, 'gray') && size(X, 3) == 3
        X = 0.2989 * X(:,:,1) + 0.5870 * X(:,:,2) + 0.1140 * X(:,:,3);
    end

    tStart = tic;
    [splitLabels, nSplit]  = splitRegions(X, params);            % fase 1
    [labels, nMerged]      = mergeRegions(X, splitLabels, params); % fase 2
    elapsed = toc(tStart);

    segmented = regionMeanImage(X, labels);

    info = struct( ...
        'params',           params, ...
        'numRegionsSplit',  nSplit, ...
        'numRegionsMerged', nMerged, ...
        'elapsedSeconds',   elapsed, ...
        'splitLabels',      splitLabels, ...
        'processedImage',   X);
end