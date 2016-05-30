function [images] = readImages(nHelen, nLFPW)
	datasetDir = [pwd '/../dataset/'];
	n = nHelen + nLFPW; 
	images = cell(n,1);
	
	% read images from Helen
	folder = [datasetDir 'helen/trainset/'];
	what = 'jpg';
	names_img = dir([folder '*.' what]);	
	for gg = 1 : nHelen
		images{gg} = imread([folder names_img(gg).name]); 
	end
	
	% read images from LFPW
	folder = [datasetDir 'lfpw/trainset/'];
	what = 'png';
	names_img = dir([folder '*.' what]);
	for gg = 1 : nLFPW
		images{nHelen + gg} = imread([folder names_img(gg).name]); 
	end
end