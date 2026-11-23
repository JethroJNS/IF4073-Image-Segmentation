function [pythonPath, isValid, info] = find_python_executable(preferredPath)
%FIND_PYTHON_EXECUTABLE Menemukan executable Python yang memiliki PyTorch & Torchvision.
%
%   [pythonPath, isValid, info] = find_python_executable()
%   [pythonPath, isValid, info] = find_python_executable(preferredPath)
%
%   Prioritas pencarian:
%     1. preferredPath (jika diberikan oleh pemanggil).
%     2. Environment variable (IF4073_PYTHON, PYTHON_EXECUTABLE, PYTHONEXECUTABLE).
%     3. Python 3.12 lokal yang terdeteksi:
%        C:\Users\LENOVO\AppData\Local\Programs\Python\Python312\python.exe
%        serta instalasi Python3* lain di %LOCALAPPDATA%\Programs\Python\.
%     4. Path executable dari pyenv MATLAB (hanya membaca string path executable,
%        TIDAK memuat PyTorch melalui in-process pyenv).
%     5. Python dari Windows PATH (where python / python).
%
%   Setiap kandidat diuji menggunakan subprocess:
%     python -c "import torch, torchvision; print(torch.__version__)"

    if nargin < 1
        preferredPath = '';
    end
    if isstring(preferredPath) && isscalar(preferredPath)
        preferredPath = char(preferredPath);
    end

    persistent cachedExecutable cachedInfo

    % Evaluasi persistent cache:
    % 1. Jika preferredPath kosong dan terdapat cache tervalidasi, gunakan cache.
    % 2. Jika preferredPath diberikan dan sama dengan executable pada cache, gunakan cache.
    % 3. Jika preferredPath berbeda, lewati cache dan jalankan validasi normal.
    cleanPref = strrep(strtrim(preferredPath), '"', '');
    if ~isempty(cachedExecutable) && isstruct(cachedInfo) && cachedInfo.isValid
        if isempty(cleanPref) || strcmpi(cleanPref, cachedExecutable)
            pythonPath = cachedExecutable;
            isValid    = true;
            info       = cachedInfo;
            return;
        end
    end

    pythonPath = '';
    isValid    = false;
    info = struct( ...
        'pythonPath',         '', ...
        'isValid',            false, ...
        'pythonVersion',      '', ...
        'torchVersion',       '', ...
        'torchvisionVersion', '', ...
        'source',             'none', ...
        'triedCandidates',    {{}}, ...
        'diagnostics',        {{}}, ...
        'errorMessage',       '');

    [candidates, sources] = collectCandidates(preferredPath);
    info.triedCandidates  = candidates;
    diagLog               = cell(1, numel(candidates));

    for k = 1:numel(candidates)
        cand = candidates{k};
        src  = sources{k};

        [ok, pyVer, torchVer, tvVer, msg] = validateCandidate(cand);
        diagLog{k} = sprintf('[%s] %s -> %s', src, cand, msg);

        if ok
            pythonPath              = cand;
            isValid                 = true;
            info.pythonPath         = cand;
            info.isValid            = true;
            info.pythonVersion      = pyVer;
            info.torchVersion       = torchVer;
            info.torchvisionVersion = tvVer;
            info.source             = src;
            info.diagnostics        = diagLog(1:k);
            info.errorMessage       = '';

            % Simpan ke persistent cache hanya hasil validasi yang sukses
            cachedExecutable = cand;
            cachedInfo       = info;
            return;
        end
    end

    info.diagnostics = diagLog;
    if isempty(candidates)
        info.errorMessage = 'Tidak ada kandidat executable Python yang ditemukan di sistem.';
    else
        info.errorMessage = sprintf( ...
            ['Tidak ditemukan Python dengan PyTorch & Torchvision yang valid.\n' ...
             'Kandidat yang telah diuji:\n  %s'], ...
            strjoin(diagLog, '\n  '));
    end
end

function [candidates, sources] = collectCandidates(preferredPath)
    candidates = {};
    sources    = {};

    % 1. Preferred path dari argumen fungsi
    if ~isempty(strtrim(preferredPath))
        [candidates, sources] = addCandidate(candidates, sources, strtrim(preferredPath), 'argument');
    end

    % 2. Environment variables
    envNames = {'IF4073_PYTHON', 'PYTHON_EXECUTABLE', 'PYTHONEXECUTABLE'};
    for i = 1:numel(envNames)
        val = strtrim(getenv(envNames{i}));
        if ~isempty(val)
            [candidates, sources] = addCandidate(candidates, sources, val, ['env:' envNames{i}]);
        end
    end

    % 3. Python 3.12 spesifik yang telah terverifikasi di mesin ini
    knownPy312 = 'C:\Users\LENOVO\AppData\Local\Programs\Python\Python312\python.exe';
    [candidates, sources] = addCandidate(candidates, sources, knownPy312, 'known_python312');

    % Scan juga %LOCALAPPDATA%\Programs\Python\Python*\python.exe (diurutkan dari versi terbaru)
    localAppData = getenv('LOCALAPPDATA');
    if ~isempty(localAppData)
        pyRoot = fullfile(localAppData, 'Programs', 'Python');
        if exist(pyRoot, 'dir') == 7
            dirs = dir(fullfile(pyRoot, 'Python*'));
            dirNames = sort({dirs([dirs.isdir]).name});
            for i = numel(dirNames):-1:1
                cand = fullfile(pyRoot, dirNames{i}, 'python.exe');
                if exist(cand, 'file') == 2
                    [candidates, sources] = addCandidate(candidates, sources, cand, 'localappdata_python');
                end
            end
        end
    end

    % 4. Path executable dari pyenv MATLAB (jika dijalankan di MATLAB)
    try
        if exist('pyenv', 'builtin') || exist('pyenv', 'file')
            pe = pyenv;
            if ~isempty(pe) && isprop(pe, 'Executable') && ~isempty(pe.Executable)
                pePath = char(pe.Executable);
                if ~isempty(strtrim(pePath))
                    [candidates, sources] = addCandidate(candidates, sources, strtrim(pePath), 'matlab_pyenv');
                end
            end
        end
    catch
        % Abaikan jika pyenv tidak tersedia (mis. di versi lama atau Octave)
    end

    % 5. Python dari Windows PATH menggunakan `where python`
    if ispc
        [st, out] = system('where python 2>nul');
        if st == 0 && ~isempty(strtrim(out))
            lines = strsplit(strtrim(out), {'\r\n', '\n', '\r'});
            for i = 1:numel(lines)
                ln = strtrim(lines{i});
                % Hindari stub Microsoft Store WindowsApps yang kosong
                if ~isempty(ln) && isempty(strfind(lower(ln), 'windowsapps')) %#ok<STREMP>
                    [candidates, sources] = addCandidate(candidates, sources, ln, 'windows_path');
                end
            end
        end
    else
        [st, out] = system('which python3 2>/dev/null');
        if st == 0 && ~isempty(strtrim(out))
            [candidates, sources] = addCandidate(candidates, sources, strtrim(out), 'unix_path');
        end
    end

    % Fallback literal 'python' di PATH
    [candidates, sources] = addCandidate(candidates, sources, 'python', 'path_fallback');
end

function [candidates, sources] = addCandidate(candidates, sources, pathStr, sourceName)
    pathStr = strrep(strtrim(pathStr), '"', '');
    if isempty(pathStr)
        return;
    end
    for i = 1:numel(candidates)
        if strcmpi(candidates{i}, pathStr)
            return;
        end
    end
    candidates{end + 1} = pathStr;
    sources{end + 1}    = sourceName;
end

function [isValid, pyVer, torchVer, tvVer, diagMsg] = validateCandidate(cand)
    isValid  = false;
    pyVer    = '';
    torchVer = '';
    tvVer    = '';

    % Jika bukan command literal 'python'/'python3', pastikan file fisiknya ada
    if ~any(strcmpi(cand, {'python', 'python3', 'python.exe', 'python3.exe'}))
        if exist(cand, 'file') ~= 2
            diagMsg = 'File executable tidak ditemukan';
            return;
        end
    end

    % Uji import torch & torchvision melalui subprocess
    pyCode = [ ...
        'import platform, torch, torchvision; ' ...
        'print(''PY_VER='' + platform.python_version()); ' ...
        'print(''TORCH_VER='' + torch.__version__); ' ...
        'print(''TV_VER='' + torchvision.__version__)' ...
    ];
    cmd = sprintf('"%s" -c "%s" 2>&1', cand, pyCode);
    [status, cmdout] = system(cmd);

    if status ~= 0
        cleanOut = strtrim(regexprep(cmdout, '[\r\n]+', ' | '));
        if isempty(cleanOut)
            cleanOut = sprintf('Exit code %d', status);
        end
        diagMsg = sprintf('Gagal import torch/torchvision (%s)', cleanOut);
        return;
    end

    pyVer    = extractToken(cmdout, 'PY_VER=');
    torchVer = extractToken(cmdout, 'TORCH_VER=');
    tvVer    = extractToken(cmdout, 'TV_VER=');

    if isempty(torchVer) || isempty(tvVer)
        diagMsg = 'Output versi torch/torchvision tidak valid';
        return;
    end

    isValid = true;
    diagMsg = sprintf('OK (Python %s, PyTorch %s, Torchvision %s)', pyVer, torchVer, tvVer);
end

function val = extractToken(textOut, prefix)
    val = '';
    lines = strsplit(textOut, {'\r\n', '\n', '\r'});
    for i = 1:numel(lines)
        ln = strtrim(lines{i});
        if strncmp(ln, prefix, numel(prefix))
            val = strtrim(ln(numel(prefix) + 1:end));
            return;
        end
    end
end
