function SE = disk_element(r)
% Structuring element berbentuk cakram (logical) dengan jari-jari r
    r = max(round(r), 0);
    [x, y] = meshgrid(-r:r, -r:r);
    SE = (x.^2 + y.^2) <= r^2 + 0.5;
end