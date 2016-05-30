
function [shape2D] = getShapeOrtho(M, V, p, R, T, a)

    % M - mean shape vector
    % V - eigenvectors
    % p - parameters of non-rigid shape
    % R - rotation matrix
    % T - translation vector (tx, ty)
    shape3D = getShape3D(M, V, p);
    shape2D = a * R(1:2,:)*shape3D' + repmat(T, 1, numel(M)/3);
    shape2D = shape2D';
end

function [shape3D] = getShape3D(M, V, params)

    shape3D = M + V * params;
    shape3D = reshape(shape3D, numel(shape3D) / 3, 3);
    
end
