function [ result,fps ] = CREST_tracking_base( opts, varargin, config, display)

global objSize;
LocGt=config.gt;
global num_channels;
num_channels=64;

% training options (SGD)
opts.train = struct([]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

output_sigma_factor = 0.1;

 gpuDevice(1);
[net1,avgImg]=initVGG16Net();

net='mdnet_vot-otb.mat';
[net_b,p] = mdnet_init(net);

nFrame=config.nFrames;
[Gt]=config.gt;
name=config.name;

result = zeros(length(nFrame), 4); result(1,:) = Gt(1,:);

scale=1;
global resize;
if strcmpi(name,'jump_1')|| strcmpi(name,'girl2_1')|| strcmpi(name,'gym_1')
  resize=80;
else 
resize=100;
end
objSize=Gt(1,3:4);
if objSize(1)*objSize(2)>resize*resize
    scale=resize/max(objSize);    
    disp('resized');
end

im1=imread(config.imgList{1});
im=imresize(im1,scale);
cell_size=4;
if size(im,3)==1
    im = cat(3, im, im, im);
    im1 = cat(3, im1, im1, im1);
end

targetLoc=round(Gt(1,:)*scale);
target_sz=[targetLoc(4) targetLoc(3)];
im_sz=size(im);
p.imgSize=size(im);
window_sz = get_search_window(target_sz, im_sz);
l1_patch_num = ceil(window_sz/ cell_size);
l1_patch_num=l1_patch_num-mod(l1_patch_num,2)+1;
cos_window = hann(l1_patch_num(1)) * hann(l1_patch_num(2))';

sz_window=size(cos_window);
pos = [targetLoc(2), targetLoc(1)] + floor(target_sz/2);
patch = get_subwindow(im, pos, window_sz);
meanImg=zeros(size(patch));
meanImg(:,:,1)=avgImg(1);
meanImg(:,:,2)=avgImg(2);
meanImg(:,:,3)=avgImg(3);
patch1 = single(patch) - meanImg;
net1.eval({'input',gpuArray(patch1)});

index=[23];
feat=cell(length(index),1);
for i=1:length(index)
    feat1 = gather(net1.vars(index(i)).value);
    feat1 = imResample(feat1, sz_window(1:2));
    feat{i} = bsxfun(@times, feat1, cos_window);                        
end
feat=feat{1};

[hf,wf,cf]=size(feat);
matrix=reshape(feat,hf*wf,cf);
coeff = pca(matrix);
coeff=coeff(:,1:num_channels);

target_sz1=ceil(target_sz/cell_size);
output_sigma = target_sz1*output_sigma_factor;
label=gaussian_shaped_labels(output_sigma, l1_patch_num);

imd=[im1];
%-------------------Display First frame----------
if display    
    figure(2);
    set(gcf,'Position',[200 300 480 320],'MenuBar','none','ToolBar','none');
    hd = imshow(imd,'initialmagnification','fit'); hold on;
    rectangle('Position', Gt(1,:), 'EdgeColor', [0 0 1], 'Linewidth', 1);    
    set(gca,'position',[0 0 1 1]);
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;   
end


tic;
cur_scale=1;
%% -------------------first frame initialization-----------
trainOpts.numEpochs=200; 

feat_=reshape(feat,hf*wf,cf);
feat_=feat_*coeff;
featPCA=reshape(feat_,hf,wf,num_channels);


 %bounding box regression 
    if(p.bbreg)        
  
        fprintf('training the bounding box regressor...\n');
        pos_examples = gen_samples('uniform_aspect', targetLoc, p.bbreg_nSamples*10, p, 0.3, 10);
        r = overlap_ratio(pos_examples,targetLoc);
        pos_examples = pos_examples(r>0.6,:);
        pos_examples = pos_examples(randsample(end,min(p.bbreg_nSamples,end)),:);
        feat_conv = mdnet_features_convX(net_b, im, pos_examples, p);
        X = permute(gather(feat_conv),[4,3,1,2]);
        X = X(:,:);
        bbox = pos_examples;
        bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
        bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
    end

[net_online]=initNet_base_4diconv(target_sz1);

trainOpts.batchSize = 1 ;
trainOpts.numSubBatches = 1 ;
trainOpts.continue = true ;
trainOpts.gpus = 1;
trainOpts.prefetch = true ;

trainOpts.expDir = opts.expDir ;
trainOpts.learningRate=5e-7;    
trainOpts.weightDecay= 1;

train=1;
imdb=[];
input={featPCA label};
featPCA1st=featPCA;
opts.train.gpus=0;
bopts.useGpu = numel(opts.train.gpus) > 0 ;
info = cnn_train_dag(net_online, imdb, input,getBatchWrapper(bopts), ...  
                     trainOpts, ...
                     'train', train, ...                     
                     opts.train) ;

                 
net_online.move('gpu');
net_online.conserveMemory = false;
%% sample active feature model
use_saf = false;
if use_saf
    sw_sz = size(label);
    saf.label_binary = single(binary_labels(filter_sz, sw_sz([2 1]), [0.3 0.7]));
    saf.weight_factor = 1.2;  %  1.2
    saf.weights = ones(1,1,num_channels);
    label_binary = saf.label_binary;
    
    feat_weight = ones(1,1,num_channels);
    
 end
%% ----------------online prediction------------------
motion_sigma_factor=0.6;
cell_size=4;
global num_update;
num_update=2;
cur=1;
feat_update=cell(num_update,1);
label_update=cell(num_update,1);
target_szU=target_sz;
for i=2:nFrame
    fprintf('Processing frame %d/%d... \n', i, nFrame);
    im1=imread(config.imgList{i});          
    im=imresize(im1,scale);    
    if size(im1,3)==1
        im = cat(3, im, im, im);
        im1 = cat(3, im1, im1, im1);
    end    
    
    patch = get_subwindow(im, pos, window_sz*cur_scale); 
    patch = imresize(patch,window_sz);
    patch1 = single(patch) - meanImg;    
    net1.eval({'input',gpuArray(patch1)});
    
    feat=cell(length(index),1);
    for j=1:length(index)
        feat1 = gather(net1.vars(index(j)).value);
        feat1 = imResample(feat1, sz_window(1:2));
        feat{j} = bsxfun(@times, feat1, cos_window);                                   
    end
    feat=feat{1};
    
    feat_=reshape(feat,hf*wf,cf);
    feat_=feat_*coeff;
    featPCA=reshape(feat_,hf,wf,num_channels);    
    feat_tmp = featPCA;
    %% adding sample active feature model
    if use_saf&0
        feat_tmp = bsxfun(@times,feat_tmp,feat_weight);
    end
    %%
      
    net_online.eval({'input1',gpuArray(feat_tmp)});  
    
    regression_map=gather(net_online.vars(net_online.getVarIndex('sum_1')).value); %lx       
             
    motion_sigma = target_sz1*motion_sigma_factor;    
    motion_map=gaussian_shaped_labels(motion_sigma, l1_patch_num);
        
    response=regression_map.*motion_map;    
    
    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
    vert_delta  = vert_delta  - ceil(hf/2);
    horiz_delta = horiz_delta - ceil(wf/2);   
        
        [target_szU, scaling]=e_scale_estimation_base(im,pos,target_szU,window_sz,...
            net1,net_online,coeff,meanImg); 
    pos = pos + cell_size * cur_scale * [vert_delta, horiz_delta];
               
    % bbox regression
    if(p.bbreg && max(response(:))>0.7)

        bbsample = [pos([2 1])-target_szU([2 1])/2  target_szU([2 1])]; %left-top(x,y,w,h) 
        feat_conv = mdnet_features_convX(net_b, im, bbsample, p);
        X_ = permute(gather(feat_conv),[4,3,1,2]);
        X_ = X_(:,:);
        bbox_ = bbsample;
        pred_boxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_);
        predictrectPosition = round(mean(pred_boxes,1));
        
        targetPosition=[predictrectPosition(2)+0.5*predictrectPosition(4) predictrectPosition(1)+0.5*predictrectPosition(3)];%(center y x)
        targetSize=[predictrectPosition(4) predictrectPosition(3)];%(h w)
        pos=targetPosition ;
        target_szU=targetSize;
    end     
                            
    targetLoc=[pos([2,1]) - target_szU([2,1])/2, target_szU([2,1])];    
    result(i,:)=round(targetLoc/scale);                  
               
    imd=[im1];
%%    -----------Display current frame-----------------
    if display   
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',imd); hold on;                                
        
        % show score map
        
        w_sz_ori = round(window_sz*cur_scale/scale);
        sw_location=imresize(regression_map,w_sz_ori);
        xs = floor(result(i,1)+result(i,3)/2) + (1:w_sz_ori(1)) - floor(w_sz_ori(1)/2);
        ys = floor(result(i,2)+result(i,4)/2) + (1:w_sz_ori(2)) - floor(w_sz_ori(2)/2);
        resp_handle = imagesc(xs, ys, sw_location); colormap hsv;
        alpha(resp_handle, 0.5);   % 0.4
        
        rectangle('Position', result(i,:), 'EdgeColor', [0 0 1], 'Linewidth', 1);                       
        set(gca,'position',[0 0 1 1]);
        text(10,10,num2str(i),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        drawnow;  
    end

    
     %% -----------Model update-------------------------                                
     labelU=circshift(label,[vert_delta,horiz_delta]);
     feat_update{cur}=featPCA;              
     label_update{cur}=labelU; 
     
     %% update binary label map based on the result of this frame
     if use_saf&0
           res_tmp = circshift(regression_map,[vert_delta,horiz_delta]);
           thres_hn = 0.5;
           saf.label_binary = label_binary;
           saf.label_binary(res_tmp<thres_hn*max(res_tmp(:)))=0; 
           if sum(sum(saf.label_binary < 0)) > 1
              [feat_weight] = SAF_generate(net_online, saf, gpuArray(feat_tmp), feat_weight);
           else
               feat_weight =feat_weight - (feat_weight-1)/5;
           end
     end
     
     
     if cur==num_update    
    
        trainOpts.batchSize = 1 ;
        trainOpts.numSubBatches = 1 ;
        trainOpts.continue = true ;
        trainOpts.gpus = 1 ;
        trainOpts.prefetch = true ;

        trainOpts.expDir = 'exp/update/' ;
        trainOpts.learningRate = 2e-9;    
        trainOpts.weightDecay= 1;
        trainOpts.numEpochs = 2;  

        train=1;
        imdb=[];
        input={feat_update featPCA1st label_update};
       
        
        opts.train.gpus=1;
        bopts.useGpu = numel(opts.train.gpus) > 0 ;
                            
        info = cnn_train_dag_update_base(net_online, imdb, input,getBatchWrapper(bopts), ...
                             trainOpts, ...  
                             'train', train, ...                     
                             opts.train) ;        
        net_online.move('gpu');   
        
        cur=1;        
    else 
        cur=cur+1;            
    end
               
end
toc;
fps=toc/nFrame;
end

function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,false,opts,'prefetch',nargout==0) ;
end
