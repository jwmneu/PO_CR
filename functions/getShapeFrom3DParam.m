function [shape2D] = getShapeFrom3DParam(M, V, p)
% get 2D landmarks from 3D shape parameters
    % M - mean shape vector
    % V - eigenvectors
    % p - parameters of non-rigid shape
    % R - rotation matrix
    % T - translation vector (tx, ty)

    % get 3D landmarks by nonrigid parameters
    shape3D = getShape3D(M, V, p(1, 7:end)');
    % transform to 2D landmarks by rigid parameters
    T = p(1, 2:3)';
    R = Euler2Rot(p(1, 4:6));
    a = p(1, 1);
    shape2D = a * R(1:2,:)*shape3D' + repmat(T, 1, numel(M)/3);
    shape2D = shape2D';
end

function [shape3D] = getShape3D(M, V, params)

    shape3D = M + V * params;
    shape3D = reshape(shape3D, numel(shape3D) / 3, 3);
    
end

function [Rot] = Euler2Rot(euler)

	rx = euler(1);
	ry = euler(2);
	rz = euler(3);

	Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
	Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
	Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
	
	Rot = Rx * Ry * Rz;
end