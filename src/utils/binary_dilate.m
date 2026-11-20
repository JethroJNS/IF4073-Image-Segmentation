function D = binary_dilate(B, SE)
    D = conv2(double(B), double(SE), 'same') > 0.5;
end