function results = edge_segmentation_all(img, params)
    if nargin < 2, params = struct(); end
    names = edge_method_list();
    for k = 1:numel(names)
        results(k) = edge_segmentation(img, names{k}, params); %#ok<AGROW>
    end
end