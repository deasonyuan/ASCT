function results=run_ASCT(seq, res_path, bSaveImage)
close all;
varargin=cell(1,2);
varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', 1)};


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

global resize ;
display=1;


img_files=seq.s_frames;
config=struct('imgList',{img_files'},'gt', seq.init_rect, 'nFrames',seq.len,'name',seq.name);

[result,fps]=tracking_base(opts,varargin,config,display);     
results.res=result;
results.fps=fps;
results.type='rect';
end