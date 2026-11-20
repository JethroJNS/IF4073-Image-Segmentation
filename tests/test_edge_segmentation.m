function test_edge_segmentation(varargin)
    opt = struct('interactive', true, 'batch', false);
    for i = 1:2:numel(varargin)
        opt.(varargin{i}) = varargin{i + 1};
    end

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(root, 'src')));

    nFail = run_unit_tests();
    if nFail == 0
        fprintf('\n>>> SEMUA UNIT TEST LULUS. Implementasi siap digunakan.\n');
    else
        fprintf('\n>>> %d UNIT TEST GAGAL. Periksa pesan di atas.\n', nFail);
    end

    if opt.batch
        run_batch(root);
    end
    if opt.interactive
        run_interactive(root);
    end
end

function nFail = run_unit_tests()
    nFail = 0;
    fprintf('=== UNIT TEST ===\n');

    nFail = nFail + check('to_gray: putih -> 1, hitam -> 0', ...
        abs(to_gray(uint8(cat(3, 255, 255, 255))) - 1) < 1e-9 && to_gray(uint8(cat(3, 0, 0, 0))) == 0);

    nFail = nFail + check('apply_kernel: ukuran keluaran = ukuran masukan (kernel 3x3 & 2x2)', ...
        isequal(size(apply_kernel(rand(7, 9), ones(3))), [7 9]) && ...
        isequal(size(apply_kernel(rand(7, 9), ones(2))), [7 9]));

    nFail = nFail + check('apply_kernel: kernel rerata pada citra konstan -> citra konstan', ...
        max(max(abs(apply_kernel(5 * ones(6), ones(3) / 9) - 5))) < 1e-12);

    nFail = nFail + check('gaussian_kernel: jumlah = 1', abs(sum(sum(gaussian_kernel(1.5))) - 1) < 1e-12);

    B = false(10); B(2:4, 2:4) = true; B(7:9, 6:9) = true;
    [~, n] = label_components(B, 8);
    nFail = nFail + check('label_components: 2 komponen', n == 2);

    ring = false(11); ring(3:9, 3:9) = true; ring(4:8, 4:8) = false;
    F = fill_holes(ring);
    nFail = nFail + check('fill_holes: cincin terisi penuh', all(all(F(3:9, 3:9))) && ~F(1, 1));

    t = otsu_threshold([0.1 * ones(1, 500), 0.9 * ones(1, 500)]);
    nFail = nFail + check('otsu_threshold: citra bimodal -> ambang di antara 0.1 dan 0.9', t >= 0.1 && t < 0.9);

    S = zeros(20, 20); S(:, 11:end) = 1;
    p = default_edge_params(struct('smoothSigma', 0));
    ops = {'Sobel', 'Prewitt', 'Roberts'};
    for k = 1:3
        det = detect_edges(S, ops{k}, p);
        cols = find(any(det.edges, 1));
        nFail = nFail + check(sprintf('%s: tepi tegak terdeteksi di sekitar kolom 10-11', ops{k}), ...
            ~isempty(cols) && all(cols >= 9 & cols <= 12) && nnz(det.edges) >= 18);
    end
    detL = detect_edges(S, 'Laplace', p);
    colsL = find(any(detL.edges, 1));
    nFail = nFail + check('Laplace: zero-crossing di sekitar kolom 10-11', ~isempty(colsL) && all(colsL >= 9 & colsL <= 12));
    detG = detect_edges(S, 'LoG', default_edge_params());
    colsG = find(any(detG.edges, 1));
    nFail = nFail + check('LoG: zero-crossing di sekitar kolom 10-11', ~isempty(colsG) && all(colsG >= 9 & colsG <= 12));
    detC = detect_edges(S, 'Canny', default_edge_params());
    colsC = find(any(detC.edges, 1));
    nFail = nFail + check('Canny: tepi di sekitar kolom 10-11', ~isempty(colsC) && all(colsC >= 8 & colsC <= 13));

    rand('seed', 1); %#ok<RAND>
    [X, Y] = meshgrid(1:128, 1:128);
    truth = (X - 64).^2 + (Y - 64).^2 <= 30^2;
    img = 0.2 + 0.6 * double(truth) + 0.02 * randn(128);
    img = min(max(img, 0), 1);

    names = edge_method_list();
    for k = 1:numel(names)
        r = edge_segmentation(uint8(255 * img), names{k});
        iou = nnz(r.mask & truth) / nnz(r.mask | truth);
        nFail = nFail + check(sprintf('Segmentasi %-8s: IoU terhadap cakram sebenarnya = %.2f (>= 0.85)', names{k}, iou), ...
            iou >= 0.85 && isequal(size(r.mask), [128 128]) && isequal(size(r.segmented), [128 128 3]));
    end

    ok = false;
    try, detect_edges(S, 'xyz'); catch, ok = true; end
    nFail = nFail + check('detect_edges: metode tak dikenal -> error', ok);
end

function f = check(name, cond)
    if cond
        fprintf('  [PASS] %s\n', name); f = 0;
    else
        fprintf('  [FAIL] %s\n', name); f = 1;
    end
end

function run_interactive(root)
    fprintf('\n=== PENGUJIAN INTERAKTIF ===\n');
    fprintf('Masukkan path gambar, atau:\n');
    fprintf('  Enter = pilih lewat dialog file | d = semua citra di folder data/ | q = keluar\n');
    while true
        s = strtrim(input('Gambar > ', 's'));
        if strcmpi(s, 'q'), break; end
        if strcmpi(s, 'd'), run_batch(root); continue; end
        if isempty(s)
            if ~usejava('desktop')
                fprintf('Dialog file tidak tersedia di mode ini, ketik path secara manual.\n'); continue;
            end
            [fn, fp] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Citra'}, 'Pilih citra', fullfile(root, 'data'));
            if isequal(fn, 0), continue; end
            s = fullfile(fp, fn);
        end
        s = strrep(strrep(s, '"', ''), '''', '');
        if ~exist(s, 'file')
            fprintf('File tidak ditemukan: %s\n', s); continue;
        end

        params = ask_params();
        names = edge_method_list();
        fprintf('Memproses (%s)...\n', s);
        results = edge_segmentation_all(s, params);
        [~, nm] = fileparts(s);
        show_edge_results(results, nm, true);
        for k = 1:numel(results)
            fprintf('  %-8s: tepi = %6d piksel | luas objek = %5.1f%% | radius penutupan = %d\n', names{k}, ...
                nnz(results(k).edges), 100 * nnz(results(k).mask) / numel(results(k).mask), results(k).radius);
        end
        save_results(results, nm, root);
    end
end

function params = ask_params()
    params = struct();
    a = strtrim(input('Pakai parameter default? [Enter = ya, n = ubah] ', 's'));
    if ~strcmpi(a, 'n'), return; end
    d = default_edge_params();
    params.smoothSigma = ask_num('smoothSigma (penghalusan)', d.smoothSigma);
    params.sensitivity = ask_num('sensitivity (pengali ambang Otsu)', d.sensitivity);
    params.logSigma    = ask_num('logSigma (LoG)', d.logSigma);
    params.zcRatio     = ask_num('zcRatio (ambang zero-crossing LoG)', d.zcRatio);
    params.numObjects  = ask_num('numObjects (jumlah objek dipertahankan)', d.numObjects);
end

function v = ask_num(label, def)
    s = strtrim(input(sprintf('  %s [%g]: ', label, def), 's'));
    v = str2double(s);
    if isempty(s) || isnan(v), v = def; end
end

function run_batch(root)
    dataDir = fullfile(root, 'data');
    files = [dir(fullfile(dataDir, '*.jpg')); dir(fullfile(dataDir, '*.png')); dir(fullfile(dataDir, '*.jpeg')); dir(fullfile(dataDir, '*.bmp'))];
    if isempty(files)
        fprintf('Tidak ada citra di %s (letakkan 8 citra BSDS500 di sana).\n', dataDir); return;
    end
    for i = 1:numel(files)
        p = fullfile(dataDir, files(i).name);
        fprintf('[%d/%d] %s\n', i, numel(files), files(i).name);
        results = edge_segmentation_all(p);
        [~, nm] = fileparts(p);
        fig = show_edge_results(results, nm, false);
        save_results(results, nm, root, fig);
        close(fig);
    end
end

function save_results(results, nm, root, fig)
    outDir = fullfile(root, 'results', 'edge_detection');
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    for k = 1:numel(results)
        m = lower(results(k).method);
        imwrite(uint8(results(k).edges) * 255, fullfile(outDir, sprintf('%s_%s_edges.png', nm, m)));
        imwrite(results(k).segmented, fullfile(outDir, sprintf('%s_%s_segmented.png', nm, m)));
        imwrite(results(k).overlay, fullfile(outDir, sprintf('%s_%s_overlay.png', nm, m)));
    end
    if nargin >= 4 && ~isempty(fig)
        print(fig, fullfile(outDir, [nm '_summary.png']), '-dpng');
    end
    fprintf('  Hasil disimpan di %s\n', outDir);
end