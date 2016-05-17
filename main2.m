clear;
Kpi = 10;
T = 3;
ridge_param = 0;
learning_rate = 0.5;
smallsize = 0;
for SIFT_scale = 15:15
	regressor_sep(SIFT_scale, Kpi, T, ridge_param, learning_rate, smallsize);
end