function res = edge_segmentation(img, method, params)
% Pipeline lengkap: citra -> deteksi tepi -> segmentasi objek.
    if nargin < 2, method = 'Sobel'; end
    if nargin < 3, params = struct(); end
    params = default_edge_params(params);

    [rgb, gray] = load_image(img);

    det = detect_edges(gray, method, params);
    seg = segment_from_edges(det.edges, params);

    mask = seg.mask;
    mask3 = repmat(mask, [1 1 3]);

    segmented = rgb;
    segmented(~mask3) = 0;                       % latar belakang -> hitam

    overlay = rgb;
    bd = mask_boundary(mask, 3);
    for ch = 1:3
        tmp = overlay(:, :, ch);
        tmp(bd) = (ch == 1) * 255;               % kontur merah
        overlay(:, :, ch) = tmp;
    end

    res.original  = rgb;
    res.gray      = gray;
    res.edges     = det.edges;
    res.response  = det.response;
    res.mask      = mask;
    res.labels    = seg.labels;
    res.segmented = segmented;
    res.overlay   = overlay;
    res.radius    = seg.radius;
    res.areaFrac  = seg.areaFrac;
    res.success   = seg.success;
    res.valid     = seg.valid;
    res.numCandidates = seg.numCandidates;
    res.method    = det.method;
    res.params    = params;
    res.info      = det.info;
end