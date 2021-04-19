function [ feat ] = mdnet_features_convX(net, img, boxes, opts)
% MDNET_FEATURES_CONVX
% Extract CNN features from bounding box regions of an input image.
%
% Hyeonseob Nam, 2015
% 

n = size(boxes,1);
ims = mdnet_extract_regions(img, boxes, opts);
nBatches = ceil(n/opts.batchSize_test);
%zFeatId = net.getVarIndex(opts.id_feat_b);
for i=1:nBatches
%     fprintf('extract batch %d/%d...\n',i,nBatches);
    
    batch = ims(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i));
    batch = gpuArray(batch);
    
    
    res = vl_simplenn(net, batch, [], [], ...
        'conserveMemory', false, ...
        'sync', false) ;
  %   net.eval({'input', batch});
   %  f = gather(net.vars(31).value);
   f = gather(res(end).x) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
  feat(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i)) = f;
end