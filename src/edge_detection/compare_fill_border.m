function T = compare_fill_border(imgDir, method, params)
% Membandingkan fillBorderObjects = false vs true pada folder citra
    if nargin < 2, method = 'Canny'; end
    if nargin < 3, params = struct(); end

    exts = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};
    files = [];
    for k = 1:numel(exts)
        files = [files; dir(fullfile(imgDir, exts{k}))]; %#ok<AGROW>
    end
    n = numel(files);
    if n == 0
        error('compare_fill_border:noImages', 'Tidak ada citra pada folder: %s', imgDir);
    end

    figure('Name', ['Perbandingan fillBorderObjects - ' method], 'NumberTitle', 'off', ...
           'Position', [40 40 180*n 600]);
    fprintf('\n%-22s | %-24s | %-24s\n', 'Citra', 'fill=false', 'fill=true');
    fprintf('%s\n', repmat('-', 1, 76));

    T = struct('name', {}, 'areaFalse', {}, 'areaTrue', {}, ...
               'validFalse', {}, 'validTrue', {}, 'successFalse', {}, 'successTrue', {});
    for i = 1:n
        path = fullfile(imgDir, files(i).name);

        pF = params; pF.fillBorderObjects = false;
        pT = params; pT.fillBorderObjects = true;
        rF = edge_segmentation(path, method, pF);
        rT = edge_segmentation(path, method, pT);

        T(i) = struct('name', files(i).name, ...
                      'areaFalse', rF.areaFrac, 'areaTrue', rT.areaFrac, ...
                      'validFalse', rF.valid, 'validTrue', rT.valid, ...
                      'successFalse', rF.success, 'successTrue', rT.success);

        fprintf('%-22s | %-24s | %-24s\n', files(i).name, fmt(rF), fmt(rT));

        subplot(3, n, i);         imshow(rF.original);
        title(files(i).name, 'Interpreter', 'none', 'FontSize', 8);
        subplot(3, n, n + i);     imshow(rF.overlay);
        title(['false ' fmt(rF)], 'FontSize', 8);
        subplot(3, n, 2*n + i);   imshow(rT.overlay);
        title(['true ' fmt(rT)], 'FontSize', 8);
    end
    drawnow;
end

function s = fmt(R)
% Luas (%), penanda status
    flag = '';
    if ~R.success
        flag = ' x';
    elseif ~R.valid
        flag = ' !';
    end
    s = sprintf('%.1f%%%s', 100 * R.areaFrac, flag);
end