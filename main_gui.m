function varargout = main_gui(varargin)
%MAIN_GUI Entry-point utama aplikasi GUI Segmentasi Citra IF4073.
%
%   main_gui()
%   fig = main_gui()
%
%   Menambahkan direktori modul:
%     - src/edge_detection
%     - src/split_merge
%     - src/utils
%     - src/gui
%   ke MATLAB path secara dinamis, lalu meluncurkan GUI terintegrasi.

    projectRoot = fileparts(mfilename('fullpath'));

    moduleDirs = { ...
        fullfile(projectRoot, 'src', 'edge_detection'), ...
        fullfile(projectRoot, 'src', 'split_merge'), ...
        fullfile(projectRoot, 'src', 'utils'), ...
        fullfile(projectRoot, 'src', 'gui')};

    for k = 1:numel(moduleDirs)
        if exist(moduleDirs{k}, 'dir') == 7
            addpath(moduleDirs{k});
        else
            warning('main_gui:missingDir', 'Direktori modul tidak ditemukan: %s', moduleDirs{k});
        end
    end

    fig = launch_segmentation_gui(varargin{:});
    if nargout > 0
        varargout{1} = fig;
    end
end
