function [lmrect] = reconstruct_lm(myShape_s0, myShape_QNonrigid, p)
	lm_pt = size(myShape_s0, 1) / 2; 
	% nonrigid parameters
	lm_center = myShape_s0+ myShape_QNonrigid * p(1, 5:end)'; 
	lm_center = reshape(lm_center , [], 2);
	
	Rot = [ cos(p(1,4)) , sin(p(1,4)); -1 * sin(p(1,4)), cos(p(1,4))];
	lmrect = p(1, 1) * lm_center * Rot + repmat(p(1, 2:3), lm_pt, 1); 
end
