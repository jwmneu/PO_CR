function [pt_pt_err_allimages, cum_err] = Compute_cum_error(pt_pt_err_image, n, titlename)
        var = 0:0.01:0.1;
	pt_pt_err_allimages = sum(pt_pt_err_image) / n;
	cum_err = zeros(size(var));
	for ii = 1:length(cum_err)
		cum_err(ii) = length(find(pt_pt_err_image<var(ii)))/length(pt_pt_err_image);
	end
	figure; hold on;
	subplot(2,1,1);
	plot(var, cum_err);
	ylim([0, 1]);
	title([titlename ' in small range']);
	grid;
	
	var = 0:0.05:0.5;
	pt_pt_err_allimages = sum(pt_pt_err_image) / n;
	cum_err = zeros(size(var));
	for ii = 1:length(cum_err)
		cum_err(ii) = length(find(pt_pt_err_image<var(ii)))/length(pt_pt_err_image);
	end
	subplot(2,1,2);
	plot(var, cum_err);
	ylim([0, 1]);
	title([titlename, ' in big range']);
	grid;
	
end
	