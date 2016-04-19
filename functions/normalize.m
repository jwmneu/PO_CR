function [normalized_features] = normalize(features)
    [r, ~] = size(features);        % r is number of images, c is number of feaures for each image. 
    maxm = max(features, [], 1);
    minm = min(features, [], 1);
    range = maxm - minm; 
    normalized_features = (features - repmat(minm, r, 1) ) ./ repmat(range, r, 1) ;
end