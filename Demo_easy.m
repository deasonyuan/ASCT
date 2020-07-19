
clear all;

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', 3)};

addpath(genpath('./base/matconvnet/matlab/'));
run vl_setupnn ;
addpath(genpath('base'));

opts.expDir = 'exp/' ;
opts.dataDir = 'exp/data/' ;
opts.modelType = 'tracking' ;
opts.sourceModelPath = 'exp/models/' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = false ;

global resize;
% display=1;
%   g=gpuDevice(1);


%% add video selection 
base_path = '/home/mmvg/yuandi/OTB100'; %lx
%ask the user for the video
video_path = choose_video(base_path);
video=video_path(length(base_path)+1:end-1);
% video = 'Skiing';

[config]=config_list(video,base_path);

display = 1;

  
result=tracking_base(opts,varargin,config,display);
% result=drt_base(opts,varargin,config,display);   % drt with calculate the scale and position simultaneously
%%
[distance_precision, area, average_center_location_error] = ...
    compute_performance_measures(result, config.gt);

fprintf('\n%f\n',area);
  
       



