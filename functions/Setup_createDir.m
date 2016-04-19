function [] = Setup_createDir(outputDir, debug)
	cd([pwd '/vlfeat-0.9.20/toolbox']);
	vl_setup
	cd '../../'

	if(exist(outputDir, 'dir') == 0)
		mkdir(outputDir);
	end
	cd(outputDir);
	if(exist('ppp', 'dir') == 0)
		mkdir('ppp');
	end
	if(exist('cum_err', 'dir') == 0)
		mkdir('cum_err');
	end
	if(exist('JPs', 'dir') == 0)
		mkdir('JPs');
	end
	if(exist('Risks', 'dir') == 0)
		mkdir('Risks');
	end
	if(exist('pt_pt_err_all', 'dir') == 0)
		mkdir('pt_pt_err_all');
	end
	if(exist('b_mat', 'dir') == 0)
		mkdir('b_mat');
	end
	if(exist('pt_pt_err', 'dir') == 0)
		mkdir('pt_pt_err');
	end
	if(exist('delta_p', 'dir') == 0)
		mkdir('delta_p');
	end
	cd('../');
	
	if debug == 1
		if(exist('debug', 'dir') == 0)
			mkdir('debug');
		end
	end
end