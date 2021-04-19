function [net_conv, opts] = mdnet_init(net)
% MDNET_INIT
% Initialize MDNet tracker.
%
% Hyeonseob Nam, 2015
% 

%% set opts
% use gpu
opts.useGpu = true;

% model def
opts.net_file = net;

% test policy
opts.batchSize_test = 256; % <- reduce it in case of out of gpu memory

% bounding box regression
opts.bbreg = false;
opts.bbreg_nSamples = 1000;

% learning policy
% opts.batchSize = 128;
% opts.batch_pos = 32;
% opts.batch_neg = 96;
% 
% % initial training policy
% opts.learningRate_init = 0.0001; % x10 for fc6
% opts.maxiter_init = 30;
% 
% opts.nPos_init = 500;
% opts.nNeg_init = 5000;
% opts.posThr_init = 0.7;
% opts.negThr_init = 0.5;

% update policy
% opts.learningRate_update = 0.0003; % x10 for fc6
% opts.maxiter_update = 10;
% 
% opts.nPos_update = 50;
% opts.nNeg_update = 200;
% opts.posThr_update = 0.7;
% opts.negThr_update = 0.3;
% 
% opts.update_interval = 10; % interval for long-term update

% data gathering policy
% opts.nFrames_long = 100; % long-term period
% opts.nFrames_short = 20; % short-term period

% cropping policy
opts.input_size = 107;
opts.crop_mode = 'wrap';
opts.crop_padding = 16;

% scaling policy
opts.scale_factor = 1.05;

% sampling policy
opts.nSamples = 256;
opts.trans_f = 0.6; % translation std: mean(width,height)*trans_f/2
opts.scale_f = 1; % scaling std: scale_factor^(scale_f/2)

% set image size
%opts.imgSize = size(image);

%% load net
net = load(opts.net_file);
if isfield(net,'net'), net = net.net; end
net_conv.layers = net.layers(1:10);
%net_fc.layers = net.layers(11:end);
clear net;

for i=1:numel(net_conv.layers)
    switch (net_conv.layers{i}.name)
        case {'conv1','conv2','conv3'}
           net_conv.layers{i}.weights{1}=net_conv.layers{i}.filters;
           net_conv.layers{i}.weights{2}= net_conv.layers{i}.biases; 
           net_conv.layers{i}.opts={};
           net_conv.layers{i}.dilate=[1 1];
        case{'relu1','relu2','relu3'}
            net_conv.layers{i}.leak=0;
        case{'pool1','pool2'}
            net_conv.layers{i}.opts={};
    end
end

if opts.useGpu
    net_conv = vl_simplenn_move(net_conv, 'gpu') ;
   % net_fc = vl_simplenn_move(net_fc, 'gpu') ;
else
    net_conv = vl_simplenn_move(net_conv, 'cpu') ;
   % net_fc = vl_simplenn_move(net_fc, 'cpu') ;
end

end