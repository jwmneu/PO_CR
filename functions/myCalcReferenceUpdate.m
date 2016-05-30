function [updated_p] = myCalcReferenceUpdate(params_delta, current_p)	% current_global is rigid parameters
	% input: params_delta: row vector, current_p: row vector
	% output : updated_p : row vector
   
	
	% Same goes for scaling, translation and nonrigid parameters
	K = size(params_delta, 2); 
	updated_p = zeros(1, K); 
	additioin_param = [1:3, 7:K]; 
	updated_p(1, additioin_param) = current_p(1, additioin_param) + params_delta(1, additioin_param); 

	% debug, update scale by multiplication
% 	updated_p(1,1) = current_p(1,1) * (1 + params_delta(1,1)); 
	
	% for rotation however, we want to make sure that the rotation matrix
	% approximation we have 
	% R' = [1, -wz, wy
	%       wz, 1, -wx
	%       -wy, wx, 1]	
	% is a legal rotation matrix, and then we combine it with current
	% rotation (through matrix multiplication) to acquire the new rotation
	
	% if delta_p for rotation is not 0, update the rotation parameters
	if sum(params_delta(1, 4:6) ~= 0) ~= 0

		R = Euler2Rot(current_p(4:6));

		wx = params_delta(1, 4);
		wy = params_delta(1, 5);
		wz = params_delta(1, 6);

		R_delta = [1, -wz, wy;
		       wz, 1, -wx;
		       -wy, wx, 1];

		% Make sure R_delta is orthonormal
		R_delta = OrthonormaliseRotation(R_delta);

		% Combine rotations
		R_final = R * R_delta;

		% Extract euler angle
		euler = Rot2Euler(R_final);	

		updated_p(1, 4:6) = euler;
	else
		updated_p(1, 4:6)  = current_p(1, 4:6);
	end
    
% 	% debug
% 	updated_p = current_p+ params_delta; 

end

function R_ortho = OrthonormaliseRotation(R)

	% U * V' is basically what we want, as it's guaranteed to be
	% orthonormal
	[U, ~, V] = svd(R);

	% We also want to make sure no reflection happened

	% get the orthogonal matrix from the initial rotation matrix
	X = U*V';

	% This makes sure that the handedness is preserved and no reflection happened
	% by making sure the determinant is 1 and not -1
	W = eye(3);
	W(3,3) = det(X);
    R_ortho = U*W*V';
end