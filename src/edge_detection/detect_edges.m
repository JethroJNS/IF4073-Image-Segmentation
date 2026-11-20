function det = detect_edges(gray, method, params)
% Dispatcher operator deteksi tepi
    if nargin < 3, params = default_edge_params(); end
    switch lower(method)
        case 'laplace', [E, R, info] = laplace_edge(gray, params);
        case 'log',     [E, R, info] = log_edge(gray, params);
        case 'sobel',   [E, R, info] = sobel_edge(gray, params);
        case 'prewitt', [E, R, info] = prewitt_edge(gray, params);
        case 'roberts', [E, R, info] = roberts_edge(gray, params);
        case 'canny',   [E, R, info] = canny_edge(gray, params);
        otherwise
            error('detect_edges:unknownMethod', ...
                  'Metode "%s" tidak dikenal. Pilihan: %s', method, strjoin(edge_method_list(), ', '));
    end
    det.edges = E;
    det.response = R;
    det.info = info;
    det.method = method;
end