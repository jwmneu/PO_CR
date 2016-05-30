function [] = mini_debug()
	clear; 
	Kpi = 10;
	KNonrigid = 7;
	load('debug_ridge/mini.mat');
	jp = ( delta_p \ b)'; 
	jp = jp(:, 2:6);
	H = jp' * jp;
	Risk = H \ jp';
	[p_updated, learned_delta_p] = update_p(Risk, p, features, Kpi, A0, KNonrigid); 
	diff_p = p - p_updated;
	d = learned_delta_p * 0.5 - delta_p;
end

function [p, delta_p] = update_p(Risk, p, features, Kpi, A0, KNonrigid)
	n = size(p, 1);
	for gg = 1:n
		delta_p(gg, :) = [0; Risk * (reshape(features(gg, :), [], 1) - A0); zeros(KNonrigid, 1)]';
		p(gg, :) = p(gg, :) + delta_p(gg, :); 
	end

end