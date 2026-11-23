function result = run_deep_learning_segmentation(imageInput, varargin)
%RUN_DEEP_LEARNING_SEGMENTATION Adapter MATLAB untuk inferensi Deep Learning CamVid 11-kelas.
%
%   result = run_deep_learning_segmentation(imagePath)
%   result = run_deep_learning_segmentation(imagePath, optionsStruct)
%   result = run_deep_learning_segmentation(imagePath, 'Name', Value, ...)
%
%   Options yang didukung:
%     - CheckpointPath   : Path ke file .pth (default: models/camvid11_full_25epochs_best.pth)
%     - OutputDir        : Folder output inferensi (default: results/gui_cache/<nama_citra>)
%     - GroundTruthPath  : Path ke label GT *_L.png (default: '' = tanpa GT;
%                          isi 'auto' untuk mencari otomatis *_L.png di data/CamVid)
%     - Device           : 'auto' (default), 'cpu', atau 'cuda'
%     - PythonExecutable : Path ke python.exe (default: '' = cari otomatis via find_python_executable)
%
%   Output struct `result` memiliki field:
%     - original        : Citra RGB asli (H x W x 3 uint8)
%     - predicted       : Mask ID kelas prediksi 0..10 (H x W uint8)
%     - predictedColor  : Mask prediksi berwarna 11 kelas (H x W x 3 uint8)
%     - overlay         : Overlay prediksi pada citra asli (H x W x 3 uint8)
%     - groundTruth     : Mask GT berwarna 11 kelas (H x W x 3 uint8), atau [] jika tidak ada GT
%     - groundTruthMask : Mask ID kelas GT (H x W uint8), atau [] jika tidak ada GT
%     - hasGroundTruth  : logical (true jika GT tersedia, false jika tidak)
%     - metrics         : struct metrik evaluasi terhadap GT (atau struct kosong jika tanpa GT)
%     - classTable      : struct array 11 kelas (classId, className, rgb, pixelCount, percentage, present, iouPercent)
%     - inferenceTime   : Waktu inferensi model dalam detik (double)
%     - totalTime       : Waktu total subprocess dalam detik (double)
%     - device          : Device yang digunakan ('cpu' atau 'cuda')
%     - outputDir       : Path absolut folder output
%     - rawJson         : Struct hasil jsondecode dari result.json
%     - pythonInfo      : Struct informasi interpreter Python dari find_python_executable

    if nargin < 1 || isempty(imageInput)
        error('deepLearning:invalidInput', ...
              'Argumen imagePath / imageInput wajib diberikan dan tidak boleh kosong.');
    end

    projectRoot = getProjectRoot();
    utilsDir    = fullfile(projectRoot, 'src', 'utils');
    if exist(utilsDir, 'dir') == 7
        addpath(utilsDir);
    end

    opts = parseAdapterOptions(projectRoot, varargin{:});

    % 1. Validasi script Python predict_single.py
    scriptPath = fullfile(projectRoot, 'src', 'deep_learning', 'predict_single.py');
    if exist(scriptPath, 'file') ~= 2
        error('deepLearning:scriptNotFound', ...
              'Script inferensi Python tidak ditemukan: %s', scriptPath);
    end

    % 2. Validasi input image (mendukung path file maupun matriks citra)
    [absImagePath, origRgb, imageStem] = resolveInputImage(imageInput, opts.OutputDir, projectRoot);

    % 3. Tentukan folder output absolut
    if isempty(opts.OutputDir)
        absOutputDir = fullfile(projectRoot, 'results', 'gui_cache', imageStem);
    else
        absOutputDir = toAbsolutePath(opts.OutputDir, projectRoot);
    end
    if exist(absOutputDir, 'dir') ~= 7
        mkdir(absOutputDir);
    end

    % 4. Validasi checkpoint model
    absCheckpointPath = toAbsolutePath(opts.CheckpointPath, projectRoot);
    if exist(absCheckpointPath, 'file') ~= 2
        error('deepLearning:checkpointNotFound', ...
              'File checkpoint model tidak ditemukan: %s', absCheckpointPath);
    end

    % 5. Validasi GroundTruthPath (jika diberikan)
    absGtPath = resolveGroundTruthPath(absImagePath, opts.GroundTruthPath, projectRoot);

    % 6. Temukan dan validasi Python executable (memastikan torch & torchvision bisa di-import)
    [pythonPath, isPyValid, pyInfo] = find_python_executable(opts.PythonExecutable);
    if ~isPyValid || isempty(pythonPath)
        error('deepLearning:pythonNotFound', ...
              'Gagal menemukan Python dengan PyTorch & Torchvision:\n%s', pyInfo.errorMessage);
    end

    % 7. Bangun command CLI dengan quoting aman untuk Windows (mengatasi spasi pada "Semester 7")
    cmd = buildSubprocessCommand( ...
        pythonPath, scriptPath, absImagePath, absCheckpointPath, absOutputDir, absGtPath, opts.Device);

    tSubStart = tic;
    [status, cmdout] = system(cmd);
    elapsedSubprocess = toc(tSubStart);

    if status ~= 0
        error('deepLearning:inferenceFailed', ...
              ['Inferensi Deep Learning gagal (exit code %d).\n' ...
               'Command : %s\n' ...
               'Pesan Python:\n%s'], ...
              status, cmd, strtrim(cmdout));
    end

    % 8. Baca dan validasi file output (result.json, pred_mask.png, pred_color.png, pred_overlay.png)
    jsonPath        = fullfile(absOutputDir, 'result.json');
    predMaskPath    = fullfile(absOutputDir, 'pred_mask.png');
    predColorPath   = fullfile(absOutputDir, 'pred_color.png');
    predOverlayPath = fullfile(absOutputDir, 'pred_overlay.png');

    requiredOutputs = {jsonPath, predMaskPath, predColorPath, predOverlayPath};
    for i = 1:numel(requiredOutputs)
        if exist(requiredOutputs{i}, 'file') ~= 2
            error('deepLearning:missingOutput', ...
                  'File output inferensi tidak ditemukan: %s\nOutput subprocess:\n%s', ...
                  requiredOutputs{i}, strtrim(cmdout));
        end
    end

    rawText = fileread(jsonPath);
    rawJson = jsondecode(rawText);

    predMask    = imread(predMaskPath);
    predColor   = imread(predColorPath);
    predOverlay = imread(predOverlayPath);

    % Pastikan tipe uint8 dan 3 kanal RGB untuk gambar warna
    predMask    = uint8(predMask);
    predColor   = ensureRgbUint8(predColor);
    predOverlay = ensureRgbUint8(predOverlay);

    % 9. Baca Ground Truth & metrik jika tersedia
    hasGT  = isfield(rawJson, 'has_ground_truth') && logical(rawJson.has_ground_truth);
    gtRgb  = [];
    gtMask = [];
    metricsStruct = struct();

    if hasGT
        gtColorPath = fullfile(absOutputDir, 'gt_color.png');
        gtMaskPath  = fullfile(absOutputDir, 'gt_mask.png');
        if exist(gtColorPath, 'file') == 2
            gtRgb = ensureRgbUint8(imread(gtColorPath));
        end
        if exist(gtMaskPath, 'file') == 2
            gtMask = uint8(imread(gtMaskPath));
        end
        if isfield(rawJson, 'ground_truth_metrics') && ~isempty(rawJson.ground_truth_metrics)
            gtm = rawJson.ground_truth_metrics;
            metricsStruct = struct( ...
                'pixelAccuracy',        double(gtm.pixel_accuracy), ...
                'pixelAccuracyPercent', double(gtm.pixel_accuracy_percent), ...
                'miou',                 double(gtm.miou), ...
                'miouPercent',          double(gtm.miou_percent), ...
                'numValidClasses',      double(gtm.num_valid_classes), ...
                'validPixels',          double(gtm.valid_pixels), ...
                'ignoredPixels',        double(gtm.ignored_pixels), ...
                'ignoredPercent',       double(gtm.ignored_percent), ...
                'perClassIoU',          gtm.per_class_iou, ...
                'perClassIoUPercent',   gtm.per_class_iou_percent, ...
                'eval360x480',          gtm.eval_360x480, ...
                'confusionMatrix',      double(gtm.confusion_matrix_11x11));
        end
    end

    % 10. Susun tabel distribusi 11 kelas (classTable)
    classTable = buildClassTable(rawJson, hasGT, metricsStruct);

    % 11. Susun struct keluaran final sesuai kontrak
    inferSec = 0;
    if isfield(rawJson, 'timing') && isfield(rawJson.timing, 'inference_seconds')
        inferSec = double(rawJson.timing.inference_seconds);
    end
    devStr = opts.Device;
    if isfield(rawJson, 'device') && ~isempty(rawJson.device)
        devStr = char(rawJson.device);
    end

    result = struct( ...
        'original',        origRgb, ...
        'predicted',       predMask, ...
        'predictedColor',  predColor, ...
        'overlay',         predOverlay, ...
        'groundTruth',     gtRgb, ...
        'groundTruthMask', gtMask, ...
        'hasGroundTruth',  hasGT, ...
        'metrics',         metricsStruct, ...
        'classTable',      classTable, ...
        'inferenceTime',   inferSec, ...
        'totalTime',       elapsedSubprocess, ...
        'device',          devStr, ...
        'outputDir',       absOutputDir, ...
        'imagePath',       absImagePath, ...
        'checkpointPath',  absCheckpointPath, ...
        'rawJson',         rawJson, ...
        'pythonInfo',      pyInfo);
end

function projectRoot = getProjectRoot()
    here = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(here));
end

function opts = parseAdapterOptions(projectRoot, varargin)
    opts = struct( ...
        'CheckpointPath',   fullfile(projectRoot, 'models', 'camvid11_full_25epochs_best.pth'), ...
        'OutputDir',        '', ...
        'GroundTruthPath',  '', ...
        'Device',           'auto', ...
        'PythonExecutable', '');

    if isempty(varargin)
        return;
    end

    if numel(varargin) == 1 && isstruct(varargin{1})
        userStruct = varargin{1};
        fNames = fieldnames(userStruct);
        for i = 1:numel(fNames)
            opts = assignOptionField(opts, fNames{i}, userStruct.(fNames{i}));
        end
    elseif mod(numel(varargin), 2) == 0
        for i = 1:2:numel(varargin)
            opts = assignOptionField(opts, varargin{i}, varargin{i + 1});
        end
    else
        error('deepLearning:invalidOptions', ...
              'Options harus berupa struct atau pasangan Name-Value.');
    end

    opts.CheckpointPath   = toCharRow(opts.CheckpointPath);
    opts.OutputDir        = toCharRow(opts.OutputDir);
    opts.GroundTruthPath  = toCharRow(opts.GroundTruthPath);
    opts.Device           = lower(strtrim(toCharRow(opts.Device)));
    opts.PythonExecutable = toCharRow(opts.PythonExecutable);

    if ~any(strcmp(opts.Device, {'auto', 'cpu', 'cuda'}))
        error('deepLearning:invalidDevice', ...
              'Device "%s" tidak valid. Gunakan ''auto'', ''cpu'', atau ''cuda''.', opts.Device);
    end
end

function opts = assignOptionField(opts, name, val)
    name = toCharRow(name);
    known = fieldnames(opts);
    idx = find(strcmpi(name, known), 1);
    if isempty(idx)
        error('deepLearning:unknownOption', 'Option tidak dikenal: %s', name);
    end
    opts.(known{idx}) = val;
end

function s = toCharRow(x)
    if isempty(x)
        s = '';
    elseif isstring(x) && isscalar(x)
        s = char(x);
    elseif ischar(x)
        s = x;
    else
        error('deepLearning:invalidString', 'Nilai parameter harus berupa char atau string.');
    end
end

function absPath = toAbsolutePath(p, projectRoot)
    p = strtrim(toCharRow(p));
    if isempty(p)
        absPath = '';
        return;
    end
    if isAbsolutePath(p)
        absPath = p;
    else
        absPath = fullfile(projectRoot, p);
    end
end

function tf = isAbsolutePath(p)
    if ispc
        tf = (numel(p) >= 2 && p(2) == ':') || (numel(p) >= 2 && p(1) == '\' && p(2) == '\');
    else
        tf = ~isempty(p) && p(1) == '/';
    end
end

function [absImagePath, origRgb, imageStem] = resolveInputImage(imageInput, outputDirOpt, projectRoot)
    if ischar(imageInput) || (isstring(imageInput) && isscalar(imageInput))
        rawPath = strtrim(char(imageInput));
        if isempty(rawPath)
            error('deepLearning:invalidImage', 'Path gambar tidak boleh kosong.');
        end
        absImagePath = toAbsolutePath(rawPath, projectRoot);
        if exist(absImagePath, 'file') ~= 2
            error('deepLearning:imageNotFound', ...
                  'File gambar input tidak ditemukan: %s', absImagePath);
        end
        if exist('load_image', 'file') == 2
            [origRgb, ~] = load_image(absImagePath);
        else
            origRgb = ensureRgbUint8(imread(absImagePath));
        end
        [~, imageStem, ~] = fileparts(absImagePath);
    elseif isnumeric(imageInput) || islogical(imageInput)
        if exist('load_image', 'file') == 2
            [origRgb, ~] = load_image(imageInput);
        else
            origRgb = ensureRgbUint8(uint8(imageInput));
        end
        imageStem = 'in_memory_image';
        if isempty(outputDirOpt)
            tmpDir = fullfile(projectRoot, 'results', 'gui_cache', imageStem);
        else
            tmpDir = toAbsolutePath(outputDirOpt, projectRoot);
        end
        if exist(tmpDir, 'dir') ~= 7
            mkdir(tmpDir);
        end
        absImagePath = fullfile(tmpDir, 'input_image.png');
        imwrite(origRgb, absImagePath);
    else
        error('deepLearning:invalidImage', ...
              'Input gambar harus berupa path file atau matriks citra.');
    end
end

function absGtPath = resolveGroundTruthPath(absImagePath, gtOpt, projectRoot)
    gtOpt = strtrim(gtOpt);
    if isempty(gtOpt)
        absGtPath = '';
        return;
    end

    if strcmpi(gtOpt, 'auto')
        % Cari otomatis pasangan <stem>_L.png di data/CamVid/<split>_labels/
        [imgDir, stem, ~] = fileparts(absImagePath);
        [parentDir, splitFolder] = fileparts(imgDir);
        cand1 = fullfile(parentDir, [splitFolder '_labels'], [stem '_L.png']);
        cand2 = fullfile(imgDir, [stem '_L.png']);
        if exist(cand1, 'file') == 2
            absGtPath = cand1;
        elseif exist(cand2, 'file') == 2
            absGtPath = cand2;
        else
            absGtPath = '';
        end
        return;
    end

    absGtPath = toAbsolutePath(gtOpt, projectRoot);
    if exist(absGtPath, 'file') ~= 2
        error('deepLearning:groundTruthNotFound', ...
              'File ground-truth tidak ditemukan: %s', absGtPath);
    end
end

function cmd = buildSubprocessCommand(pyExe, scriptPath, imgPath, ckptPath, outDir, gtPath, deviceStr)
    if isempty(gtPath)
        cmd = sprintf('"%s" "%s" --image "%s" --checkpoint "%s" --output-dir "%s" --device "%s" 2>&1', ...
                      pyExe, scriptPath, imgPath, ckptPath, outDir, deviceStr);
    else
        cmd = sprintf('"%s" "%s" --image "%s" --checkpoint "%s" --output-dir "%s" --ground-truth "%s" --device "%s" 2>&1', ...
                      pyExe, scriptPath, imgPath, ckptPath, outDir, gtPath, deviceStr);
    end
end

function rgb = ensureRgbUint8(A)
    if isa(A, 'uint16')
        A = uint8(double(A) / 257);
    elseif ~isa(A, 'uint8')
        A = uint8(A);
    end
    if ndims(A) == 2
        rgb = repmat(A, [1 1 3]);
    elseif size(A, 3) >= 3
        rgb = A(:, :, 1:3);
    else
        rgb = repmat(A(:, :, 1), [1 1 3]);
    end
end

function classTable = buildClassTable(rawJson, hasGT, metricsStruct)
    dist = rawJson.class_distribution;
    nClasses = numel(dist);
    classTable = repmat(struct( ...
        'classId',    0, ...
        'className',  '', ...
        'rgb',        [0 0 0], ...
        'pixelCount', 0, ...
        'percentage', 0.0, ...
        'present',    false, ...
        'iouPercent', NaN), nClasses, 1);

    for i = 1:nClasses
        if iscell(dist)
            item = dist{i};
        else
            item = dist(i);
        end
        cName = char(item.class_name);
        iouVal = NaN;
        if hasGT && isfield(metricsStruct, 'perClassIoUPercent') && ...
                isfield(metricsStruct.perClassIoUPercent, cName)
            iouVal = double(metricsStruct.perClassIoUPercent.(cName));
        end
        classTable(i).classId    = double(item.class_id);
        classTable(i).className  = cName;
        classTable(i).rgb        = double(item.rgb(:)');
        classTable(i).pixelCount = double(item.pixel_count);
        classTable(i).percentage = double(item.percentage);
        classTable(i).present    = logical(item.present);
        classTable(i).iouPercent = iouVal;
    end
end
