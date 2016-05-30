function [pt_pt_err_allimages, cum_err1] = Compute_cum_error(pt_pt_err_image, n, outputDir, titlename, makeplot)
% 	outputDir = 'Results/plots/';
% 	if(exist(outputDir, 'dir') == 0)
% 		mkdir(outputDir);
% 	end
        var = 0:0.001:0.07;
	pt_pt_err_allimages = sum(pt_pt_err_image) / n;
	cum_err1 = zeros(size(var));
	for ii = 1:length(cum_err1)
		cum_err1(ii) = length(find(pt_pt_err_image<var(ii)))/length(pt_pt_err_image);
	end
	if makeplot == 1
		figure; hold on;
		set(gcf,'color','w');
		subplot(2,1,1);
		plot(var, cum_err1);
		ylim([0, 1]);
		title([titlename ' in small range']);
		grid;
	end
	
	var = 0:0.001:0.5;
	pt_pt_err_allimages = sum(pt_pt_err_image) / n;
	cum_err = zeros(size(var));
	for ii = 1:length(cum_err)
		cum_err(ii) = length(find(pt_pt_err_image<var(ii)))/length(pt_pt_err_image);
	end
	if makeplot == 1
		subplot(2,1,2);
		plot(var, cum_err);
		ylim([0, 1]);
		title([titlename, ' in big range']);
		grid;
		savefig(strcat(outputDir, 'cum_error'));
	end
end