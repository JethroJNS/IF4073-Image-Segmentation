function test_split_merge(mode)
    if nargin < 1, mode = 'all'; end
    mode = lower(mode);

    rootDir = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(rootDir, 'src')));

    if any(strcmp(mode, {'all', 'auto'}))
        report = runAutomatedTests();
        fprintf('\n==== HASIL: %d lulus, %d gagal ====\n', report.passed, report.failed);
        if report.failed > 0
            warning('Ada pengujian yang gagal, periksa log di atas.');
        end
    end
    if any(strcmp(mode, {'all', 'interactive'}))
        runInteractive(rootDir);
    end
end

function report = runAutomatedTests()
    report = struct('passed', 0, 'failed', 0);
    fprintf('==== UJI OTOMATIS SPLIT AND MERGE ====\n');
    seedRandom(7);

    img = uint8(100 * ones(48, 48, 3));
    [L, ~, info] = splitMergeSegmentation(img);
    report = check(report, 'Citra seragam menghasilkan 1 region', ...
                   max(L(:)) == 1 && info.numRegionsSplit == 1);

    img = uint8([50 * ones(64, 32), 200 * ones(64, 32)]);
    [L, seg] = splitMergeSegmentation(img);
    report = check(report, 'Dua separuh: tepat 2 region', max(L(:)) == 2);
    report = check(report, 'Dua separuh: batas region tepat di tengah', ...
                   all(all(L(:,1:32) == L(1,1))) && all(all(L(:,33:64) == L(1,64))) ...
                   && L(1,1) ~= L(1,64));
    report = check(report, 'Dua separuh: citra hasil identik dengan asli', isequal(seg, img));

    q = zeros(64, 64, 3);
    q(1:32,  1:32,  :) = repmat(reshape([255 0 0],   1, 1, 3), 32, 32);
    q(1:32,  33:64, :) = repmat(reshape([0 255 0],   1, 1, 3), 32, 32);
    q(33:64, 1:32,  :) = repmat(reshape([0 0 255],   1, 1, 3), 32, 32);
    q(33:64, 33:64, :) = repmat(reshape([255 255 0], 1, 1, 3), 32, 32);
    q8 = uint8(q);
    [L, seg] = splitMergeSegmentation(q8);
    report = check(report, 'Empat kuadran warna: 4 region', max(L(:)) == 4);
    report = check(report, 'Empat kuadran warna: citra hasil identik', isequal(seg, q8));

    base = [60 * ones(64, 25), 190 * ones(64, 39)];
    noisy = uint8(min(max(base + 4 * randn(64, 64), 0), 255));
    [L, ~, ~] = splitMergeSegmentation(noisy, 'SplitThreshold', 15, 'MinBlockSize', 1, ...
                                       'MergeThreshold', 25, 'MinRegionSize', 10);
    report = check(report, 'Citra berderau: 2 region', max(L(:)) == 2);
    report = check(report, 'Citra berderau: batas tepat di kolom 25/26', ...
                   all(all(L(:,1:25) == L(1,1))) && all(all(L(:,26:end) == L(1,end))) ...
                   && L(1,1) ~= L(1,end));
    report = check(report, 'Citra berderau: semua region terhubung', allRegionsConnected(L));

    rnd = uint8(floor(256 * rand(37, 53, 3)));
    [L, seg, info] = splitMergeSegmentation(rnd);
    nL = numel(unique(L));
    report = check(report, 'Ukuran 37x53: ukuran label & hasil benar', ...
                   isequal(size(L), [37 53]) && isequal(size(seg), [37 53 3]));
    report = check(report, 'Label berurutan 1..M, mencakup seluruh piksel', ...
                   min(L(:)) == 1 && max(L(:)) == nL && nL == info.numRegionsMerged);
    report = check(report, 'Merge tidak menambah region', ...
                   info.numRegionsMerged <= info.numRegionsSplit);
    report = check(report, 'Citra acak: semua region terhubung', allRegionsConnected(L));

    [xx, ~] = meshgrid(linspace(0, 255, 128), 1:96);
    grad = xx + 3 * randn(96, 128);
    grad(30:60, 40:80) = 250;                       % objek terang
    grad = uint8(min(max(grad, 0), 255));
    splitCounts = zeros(1, 3);
    thr = [5 20 60];
    for k = 1:3
        [~, ~, inf2] = splitMergeSegmentation(grad, 'SplitThreshold', thr(k));
        splitCounts(k) = inf2.numRegionsSplit;
    end
    report = check(report, 'SplitThreshold naik -> region hasil split tidak bertambah', ...
                   splitCounts(1) >= splitCounts(2) && splitCounts(2) >= splitCounts(3));
    mt = [2 120];
    mc = zeros(1, 2);
    for k = 1:2
        [~, ~, inf2] = splitMergeSegmentation(grad, 'MergeThreshold', mt(k));
        mc(k) = inf2.numRegionsMerged;
    end
    report = check(report, 'MergeThreshold besar -> region akhir lebih sedikit', mc(1) > mc(2));

    g = uint8(floor(256 * rand(37, 53)));
    L = splitMergeSegmentation(g, 'SplitThreshold', 20, 'MergeThreshold', 10, ...
                               'MinRegionSize', 25);
    cnt = accumarray(L(:), 1);
    report = check(report, 'MinRegionSize: tidak ada region lebih kecil dari batas', ...
                   numel(cnt) == 1 || min(cnt) >= 25);

    [~, ~, iA] = splitMergeSegmentation(grad, 'MergeThreshold', 120);
    [~, ~, iB] = splitMergeSegmentation(grad, 'MergeThreshold', 120, 'MergeStdMax', 10);
    report = check(report, 'MergeStdMax kecil -> region akhir lebih banyak/sama', ...
                   iB.numRegionsMerged >= iA.numRegionsMerged);

    [L1] = splitMergeSegmentation(q8);
    [L2] = splitMergeSegmentation(q / 255);                 % double 0..1
    [L3] = splitMergeSegmentation(uint16(q) * 257);         % uint16
    report = check(report, 'Tipe uint8 / double 0-1 / uint16 memberi hasil sama', ...
                   isequal(L1, L2) && isequal(L1, L3));

    [~, segG] = splitMergeSegmentation(q8, 'ColorMode', 'gray');
    report = check(report, 'ColorMode gray menghasilkan 1 kanal', ismatrix(segG));

    pStd = splitMergeParams('SplitThreshold', 12);
    pRng = splitMergeParams('SplitCriterion', 'range', 'SplitThreshold', 12);
    blk = zeros(8, 8); blk(:, 5:8) = 255;
    report = check(report, 'isHomogeneous: blok konstan = true', ...
                   isHomogeneous(5 * ones(8, 8), pStd));
    report = check(report, 'isHomogeneous: blok 0/255 = false (std & range)', ...
                   ~isHomogeneous(blk, pStd) && ~isHomogeneous(blk, pRng));
    blk2 = 100 + 10 * (rand(8, 8) > 0.5);                  % rentang 10 <= 12
    report = check(report, 'isHomogeneous: range 10 <= 12 = true', isHomogeneous(blk2, pRng));

    report = check(report, 'Error: citra kosong',          throwsError(@() splitMergeSegmentation([])));
    report = check(report, 'Error: threshold negatif',     throwsError(@() splitMergeSegmentation(q8, 'SplitThreshold', -1)));
    report = check(report, 'Error: parameter tidak dikenal', throwsError(@() splitMergeSegmentation(q8, 'Foo', 1)));
    report = check(report, 'Error: MinBlockSize non-integer', throwsError(@() splitMergeSegmentation(q8, 'MinBlockSize', 2.5)));
    report = check(report, 'Error: argumen ganjil',        throwsError(@() splitMergeSegmentation(q8, 'SplitThreshold')));
end

function report = check(report, name, cond)
    if cond
        report.passed = report.passed + 1;
        fprintf('  [PASS] %s\n', name);
    else
        report.failed = report.failed + 1;
        fprintf('  [FAIL] %s\n', name);
    end
end

function tf = throwsError(fn)
    tf = false;
    try
        fn();
    catch
        tf = true;
    end
end

function seedRandom(s)
    if exist('rng', 'file') || exist('rng', 'builtin')
        rng(s);
    else
        rand('seed', s); randn('seed', s); %#ok<RAND>
    end
end

function tf = allRegionsConnected(L)
    tf = true;
    for k = 1:max(L(:))
        mask = (L == k);
        idx = find(mask, 1);
        reached = false(size(mask));
        reached(idx) = true;
        while true
            grow = reached;
            grow(1:end-1, :) = grow(1:end-1, :) | reached(2:end, :);
            grow(2:end, :)   = grow(2:end, :)   | reached(1:end-1, :);
            grow(:, 1:end-1) = grow(:, 1:end-1) | reached(:, 2:end);
            grow(:, 2:end)   = grow(:, 2:end)   | reached(:, 1:end-1);
            grow = grow & mask;
            if isequal(grow, reached), break; end
            reached = grow;
        end
        if ~isequal(reached, mask)
            tf = false;
            return;
        end
    end
end

function runInteractive(rootDir)
    dataDir    = fullfile(rootDir, 'data');
    resultsDir = fullfile(rootDir, 'results', 'split_merge');
    fprintf('\n==== MODE INTERAKTIF SPLIT AND MERGE ====\n');

    while true
        imgPath = askImagePath(dataDir);
        if isempty(imgPath), break; end

        try
            [img, map] = imread(imgPath);
            if ~isempty(map)
                img = uint8(ind2rgb(img, map) * 255);
            end
        catch err
            fprintf('Gagal membaca gambar: %s\n', err.message);
            continue;
        end
        img = limitSize(img, 600);
        [~, name] = fileparts(imgPath);

        params = askParameters();
        fprintf('\nMenjalankan split and merge (%dx%d)...\n', size(img,1), size(img,2));
        [labels, seg, info] = splitMergeSegmentation(img, params);
        fprintf('  Region setelah split : %d\n', info.numRegionsSplit);
        fprintf('  Region setelah merge : %d\n', info.numRegionsMerged);
        fprintf('  Waktu                : %.2f detik\n', info.elapsedSeconds);

        showResults(img, labels, seg, info, name);

        if askYesNo('Tampilkan studi pengaruh parameter?', false)
            showParameterStudy(img, params, name);
        end
        if askYesNo('Simpan hasil ke results/split_merge/?', false)
            saveResults(resultsDir, name, img, labels, seg);
        end
        if ~askYesNo('Uji gambar lain?', true), break; end
    end
    fprintf('Selesai.\n');
end

function imgPath = askImagePath(dataDir)
    imgPath = '';
    while true
        s = strtrim(input('\nPath gambar (Enter = dialog pilih file, q = keluar): ', 's'));
        s = strrep(strrep(s, '"', ''), '''', '');
        if strcmpi(s, 'q'), return; end
        if isempty(s)
            if usejava('desktop')
                startDir = pwd;
                if exist(dataDir, 'dir'), startDir = dataDir; end
                [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.gif', ...
                                    'Citra (jpg, png, bmp, tif, gif)'}, ...
                                   'Pilih citra uji', [startDir filesep]);
                if isequal(f, 0), return; end
                imgPath = fullfile(p, f);
                return;
            else
                fprintf('Dialog tidak tersedia; ketik path gambar secara manual.\n');
                continue;
            end
        end
        if exist(s, 'file') == 2
            imgPath = s;
            return;
        end
        fprintf('File tidak ditemukan: %s\n', s);
    end
end

function params = askParameters()
    params = splitMergeParams();
    if askYesNo('Gunakan parameter default?', true), return; end
    params.SplitCriterion = askChoice('Kriteria split', {'std', 'range'}, params.SplitCriterion);
    if strcmp(params.SplitCriterion, 'range'), params.SplitThreshold = 36; end
    params.SplitThreshold = askNumber('SplitThreshold', params.SplitThreshold);
    params.MinBlockSize   = askNumber('MinBlockSize (piksel)', params.MinBlockSize);
    params.MergeThreshold = askNumber('MergeThreshold', params.MergeThreshold);
    params.MinRegionSize  = askNumber('MinRegionSize (piksel)', params.MinRegionSize);
    params.ColorMode      = askChoice('ColorMode', {'color', 'gray'}, params.ColorMode);
    try
        params = splitMergeParams(params);
    catch err
        fprintf('Parameter tidak valid (%s). Memakai default.\n', err.message);
        params = splitMergeParams();
    end
end

function v = askNumber(label, def)
    s = strtrim(input(sprintf('  %s [%g]: ', label, def), 's'));
    v = str2double(s);
    if isempty(s) || isnan(v), v = def; end
end

function c = askChoice(label, options, def)
    s = lower(strtrim(input(sprintf('  %s (%s) [%s]: ', label, strjoin(options, '/'), def), 's')));
    if any(strcmp(s, options)), c = s; else c = def; end
end

function tf = askYesNo(question, defYes)
    if defYes, hint = 'Y/n'; else hint = 'y/N'; end
    s = lower(strtrim(input(sprintf('%s [%s]: ', question, hint), 's')));
    if isempty(s), tf = defYes; else tf = (s(1) == 'y'); end
end

function img = limitSize(img, maxDim)
    s = ceil(max(size(img, 1), size(img, 2)) / maxDim);
    if s > 1
        fprintf('Gambar besar, di-subsample dengan faktor %d agar cepat.\n', s);
        img = img(1:s:end, 1:s:end, :);
    end
end

function rgb = toDisplay(X)
    u = uint8(round(X));
    if size(u, 3) == 1, u = cat(3, u, u, u); end
    rgb = u;
end

function showResults(img, labels, seg, info, name)
    orig      = toDisplay(imageToDouble255(img));
    splitMean = regionMeanImage(info.processedImage, info.splitLabels);
    figure('Name', ['Split and Merge - ' name], 'NumberTitle', 'off');

    subplot(2, 3, 1); image(orig); axis image off; title('Citra asli');
    subplot(2, 3, 2);
    image(overlayBoundaries(orig, labelBoundaries(info.splitLabels), [255 255 0]));
    axis image off; title(sprintf('Split (quadtree): %d blok', info.numRegionsSplit));
    subplot(2, 3, 3); image(toDisplay(double(splitMean))); axis image off;
    title('Rata-rata warna setelah split');
    subplot(2, 3, 4); image(colorizeLabels(labels)); axis image off;
    title(sprintf('Label region akhir: %d', info.numRegionsMerged));
    subplot(2, 3, 5); image(toDisplay(double(seg))); axis image off;
    title('Hasil segmentasi (merge)');
    subplot(2, 3, 6);
    image(overlayBoundaries(orig, labelBoundaries(labels), [255 0 0]));
    axis image off; title('Batas region pada citra asli');
    drawnow;
end

function showParameterStudy(img, params, name)
    if strcmp(params.SplitCriterion, 'range'), sc = 3; else sc = 1; end
    splitVals = [5 10 20 40] * sc;
    mergeVals = [5 15 30 60];
    figure('Name', ['Pengaruh parameter - ' name], 'NumberTitle', 'off');
    for k = 1:4
        p = params; p.SplitThreshold = splitVals(k);
        [~, s1, i1] = splitMergeSegmentation(img, p);
        subplot(2, 4, k); image(toDisplay(double(s1))); axis image off;
        title(sprintf('SplitThr=%g, %d region', splitVals(k), i1.numRegionsMerged));

        p = params; p.MergeThreshold = mergeVals(k);
        [~, s2, i2] = splitMergeSegmentation(img, p);
        subplot(2, 4, 4 + k); image(toDisplay(double(s2))); axis image off;
        title(sprintf('MergeThr=%g, %d region', mergeVals(k), i2.numRegionsMerged));
    end
    drawnow;
end

function saveResults(resultsDir, name, img, labels, seg)
    if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
    orig = toDisplay(imageToDouble255(img));
    imwrite(seg, fullfile(resultsDir, [name '_segmented.png']));
    imwrite(colorizeLabels(labels), fullfile(resultsDir, [name '_labels.png']));
    imwrite(overlayBoundaries(orig, labelBoundaries(labels)), ...
            fullfile(resultsDir, [name '_boundaries.png']));
    fprintf('Hasil disimpan di %s\n', resultsDir);
end