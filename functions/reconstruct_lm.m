function [lmrect] = reconstruct_lm(p_nonrigid, p_rigid)
	modelDir = 'matfiles/';
	myShape = load([modelDir 'myShape.mat']); 
	myShape = myShape.myShape;
	
	% nonrigid parameters
	lm_center = myShape.s0 + myShape.QNonrigid * p_nonrigid'; 
	lm_center = reshape(lm_center , [], 2);
	
	Rot = [ cos(p_rigid(1,4)) , sin(p_rigid(1,4)); -1 * sin(p_rigid(1,4)), cos(p_rigid(1,4))];
	lmrect = p_rigid(1, 1) * lm_center * Rot + repmat(p_rigid(1, 2:3), 5, 1); 
end
