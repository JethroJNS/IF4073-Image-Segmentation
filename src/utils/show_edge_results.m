function fig = show_edge_results(results, titleText, visible)
% Menampilkan hasil: citra asli, deteksi tepi, kontur, dan objek tersegmentasi
    if nargin < 2, titleText = ''; end
    if nargin < 3, visible = true; end
    n = numel(results);

    if visible, vis = 'on'; else, vis = 'off'; end
    fig = figure('Name', ['Segmentasi Deteksi Tepi ' titleText], ...
                 'NumberTitle', 'off', 'Visible', vis, 'Position', [60 60 220*n 780]);

    rowNames = {'Citra asli', 'Deteksi tepi', 'Kontur segmentasi', 'Objek tersegmentasi'};
    for k = 1:n
        R = results(k);
        panels = {R.original, uint8(R.edges) * 255, R.overlay, R.segmented};
        for row = 1:4
            subplot(4, n, (row - 1) * n + k);
            if row == 2
                imshow(panels{row}, [0 255]);
            else
                imshow(panels{row});
            end
            if row == 1, title(R.method, 'FontWeight', 'bold'); end
            if k == 1
                % ylabel tidak tampil pada axes imshow -> pakai text di sisi kiri axes
                text(-0.03, 0.5, rowNames{row}, 'Units', 'normalized', ...
                     'Rotation', 90, 'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'bottom', 'FontSize', 9, ...
                     'Interpreter', 'none');
            end
        end
    end
    try
        sgtitle(titleText, 'Interpreter', 'none');
    catch
    end
    drawnow;
end