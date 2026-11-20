function E = binary_erode(B, SE)
    E = ~(conv2(double(~B), double(SE), 'same') > 0.5);
end