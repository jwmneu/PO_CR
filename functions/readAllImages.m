function [TR_images] = readAllImages()
	[images_helen] = readAllImagesFromHelen();
	[images_lfpw] = readAllImagesFromLFPW();
	TR_images = {};
	TR_images = cat(1, TR_images, images_helen'); 
	TR_images = cat(1, TR_images, images_lfpw');
end

function [images] = readAllImagesFromHelen()
	datasetDir = [pwd '/../dataset/'];
	folder = [datasetDir 'helen/trainset/'];
	what = 'jpg';
	names_img = dir([folder '*.' what]);
	for gg = 1 : 2000
		images{gg} = imread([folder names_img(gg).name]); 
	end
end

function [images] = readAllImagesFromLFPW()
	datasetDir = [pwd '/../dataset/'];
	folder = [datasetDir 'lfpw/trainset/'];
	what = 'png';
	names_img = dir([folder '*.' what]);
	for gg = 1 : 811
		images{gg} = imread([folder names_img(gg).name]); 
	end
end