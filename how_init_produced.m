%% Main script illustrating how initializations were produced
clear; clc; close all;

%% load models
load shape_model;
load std_detector.mat

%% select database 
% IBUG
folder = './test_data/iBUG/';
what  = 'jpg';

%% adjust noise level
std_noise = 5;
std_init = std_noise*std_dr;

%% Select image
gg = 1;
names1 = dir([folder '*.' what]);
names2 = dir([folder '*.pts']);
input_image = imread([folder names1(gg).name]); % input image must be in [0,255]!!
num_of_pts = 68; % num of landmarks in the annotations
pts = read_shape([folder names2(gg).name], num_of_pts);

%% ground_truth
gt_s = (pts-1);
gt_s = reshape(gt_s, 68, 2);
face_size = (max(gt_s(:,1)) - min(gt_s(:,1)) + max(gt_s(:,2)) - min(gt_s(:,2)))/2;

%% initialization
init_shape = gt_s;
[~,~,T] = procrustes(shape.s0, init_shape);
scl = 1/T.b;
init_shape = init_shape*(1/scl);
input_image = imresize(input_image, (1/scl));

r = shape.Q'*(init_shape(:) - shape.s0(:));
r(1) = r(1) + std_init(1).*rand(1) - std_init(1)/2;
r(2) = 0;
r(3) = r(3) + std_init(3).*rand(1) - std_init(3)/2;
r(4) = r(4) + std_init(4).*rand(1) - std_init(4)/2;
init_shape = shape.Q*r + shape.s0(:);
init_shape = reshape(init_shape, [], 2);

figure;imagesc(input_image); colormap(gray); hold on; plot(init_shape(:,1), init_shape(:,2), 'o');     
