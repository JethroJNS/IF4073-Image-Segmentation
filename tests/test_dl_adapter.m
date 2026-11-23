function test_dl_adapter()
%TEST_DL_ADAPTER Unit test untuk find_python_executable.m dan run_deep_learning_segmentation.m.
%
%   Menguji 6 skenario wajib:
%     1. Python executable ditemukan dan PyTorch + Torchvision tervalidasi.
%     2. Inference satu gambar CamVid dengan Ground Truth (GT).
%     3. Inference satu gambar tanpa Ground Truth (GT).
%     4. Penanganan path gambar tidak valid (melempar error informatif).
%     5. Penanganan path checkpoint tidak valid (melempar error informatif).
%     6. Output struct memiliki seluruh field yang ditentukan sesuai kontrak.

    rootDir = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(rootDir, 'src')));

    nFail = 0;
    fprintf('=== PENGUJIAN MATLAB-PYTHON DEEP LEARNING ADAPTER ===\n');

    % 1. Uji pencarian & validasi Python executable
    [pyPath, isValid, pyInfo] = find_python_executable();
    nFail = nFail + check('1. find_python_executable: menemukan Python dengan PyTorch & Torchvision', ...
        isValid && ~isempty(pyPath) && ~isempty(pyInfo.torchVersion) && ~isempty(pyInfo.torchvisionVersion));
    if isValid
        fprintf('     Python Path : %s\n', pyPath);
        fprintf('     Versi       : Python %s | PyTorch %s | Torchvision %s (source: %s)\n', ...
            pyInfo.pythonVersion, pyInfo.torchVersion, pyInfo.torchvisionVersion, pyInfo.source);
    end

    % 2. Uji inference satu gambar CamVid DENGAN Ground Truth
    imgWithGT = fullfile(rootDir, 'data', 'CamVid', 'test', '0016E5_08131.png');
    lblWithGT = fullfile(rootDir, 'data', 'CamVid', 'test_labels', '0016E5_08131_L.png');
    outDirGT  = fullfile(rootDir, 'results', 'gui_cache', 'test_adapter_with_gt');

    resGT = run_deep_learning_segmentation(imgWithGT, ...
        'GroundTruthPath', lblWithGT, ...
        'OutputDir', outDirGT);

    nFail = nFail + check('2. Inference dengan GT: hasGroundTruth == true & groundTruth tidak kosong', ...
        resGT.hasGroundTruth && ~isempty(resGT.groundTruth) && isequal(size(resGT.groundTruth), [720 960 3]));
    nFail = nFail + check('2b. Inference dengan GT: Pixel Accuracy & mIoU valid (> 0)', ...
        isfield(resGT.metrics, 'pixelAccuracyPercent') && resGT.metrics.pixelAccuracyPercent > 90 && ...
        isfield(resGT.metrics, 'miouPercent') && resGT.metrics.miouPercent > 70);

    % 3. Uji inference satu gambar TANPA Ground Truth
    imgNoGT  = fullfile(rootDir, 'data', 'CamVid', 'test', '0001TP_007170.png');
    outNoGT  = fullfile(rootDir, 'results', 'gui_cache', 'test_adapter_no_gt');

    resNoGT = run_deep_learning_segmentation(imgNoGT, 'OutputDir', outNoGT);
    nFail = nFail + check('3. Inference tanpa GT: hasGroundTruth == false & groundTruth kosong ([])', ...
        ~resNoGT.hasGroundTruth && isempty(resNoGT.groundTruth));

    % 4. Uji penanganan path gambar tidak valid
    errImgCaught = false;
    try
        run_deep_learning_segmentation(fullfile(rootDir, 'data', 'tidak_ada_gambar.png'));
    catch err
        errImgCaught = ~isempty(strfind(err.identifier, 'imageNotFound')); %#ok<STREMP>
    end
    nFail = nFail + check('4. Path gambar tidak valid -> melempar error deepLearning:imageNotFound', errImgCaught);

    % 5. Uji penanganan path checkpoint tidak valid
    errCkptCaught = false;
    try
        run_deep_learning_segmentation(imgNoGT, ...
            'CheckpointPath', fullfile(rootDir, 'models', 'checkpoint_palsu_tidak_ada.pth'));
    catch err
        errCkptCaught = ~isempty(strfind(err.identifier, 'checkpointNotFound')); %#ok<STREMP>
    end
    nFail = nFail + check('5. Path checkpoint tidak valid -> melempar error deepLearning:checkpointNotFound', errCkptCaught);

    % 6. Verifikasi seluruh field wajib pada output struct result
    requiredFields = { ...
        'original', 'predicted', 'predictedColor', 'overlay', ...
        'groundTruth', 'hasGroundTruth', 'metrics', 'classTable', ...
        'inferenceTime', 'device', 'outputDir'};
    hasAllFieldsGT   = all(isfield(resGT, requiredFields));
    hasAllFieldsNoGT = all(isfield(resNoGT, requiredFields));
    dimsMatch = isequal(size(resGT.original), [720 960 3]) && ...
                isequal(size(resGT.predicted), [720 960]) && ...
                isequal(size(resGT.predictedColor), [720 960 3]) && ...
                isequal(size(resGT.overlay), [720 960 3]) && ...
                numel(resGT.classTable) == 11;

    nFail = nFail + check('6. Output struct memiliki seluruh 11 field wajib & dimensi konsisten', ...
        hasAllFieldsGT && hasAllFieldsNoGT && dimsMatch);

    if nFail == 0
        fprintf('\n>>> SEMUA UNIT TEST MATLAB-PYTHON ADAPTER LULUS. <<<\n');
    else
        fprintf('\n>>> %d UNIT TEST GAGAL. <<<\n', nFail);
    end
end

function f = check(name, cond)
    if cond
        fprintf('  [PASS] %s\n', name);
        f = 0;
    else
        fprintf('  [FAIL] %s\n', name);
        f = 1;
    end
end
