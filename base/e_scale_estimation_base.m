function [target_szU,scaling] = e_scale_estimation_base(im,pos,target_sz,window_szo,...
    net1,net_online,coeff,avgImg)

num_channels=64;
threshold=0.15; %0.15

[h,w,~]=size(im);
im_sz=[h w];

cell_size=4;
l1_patch_num = ceil(window_szo/ cell_size);
l1_patch_num=l1_patch_num-mod(l1_patch_num,2)+1;
cos_window = hann(l1_patch_num(1)) * hann(l1_patch_num(2))';
sz_window=size(cos_window);

%%  ---------scale refinement------------
% scale=[1 0.97 0.95 1.03 1.05];
scale=[1 0.95 1.05];
scale_num = numel(scale);
% value=zeros(3,1);

target_sz1 = round(bsxfun(@times,target_sz,scale'));
window_sz=e_get_search_window(target_sz1,im_sz);

patches = cell(1,1,1,scale_num);
for i=1:scale_num
    patches{i} = get_subwindow(im, pos, window_sz(i,:));
end
 patches= cell2mat(cellfun(@(pat) imresize(pat,window_szo),patches,'UniformOutput',false));
 patches = double(patches);
 % patch1=double(imresize(patch,window_szo));
 
    net1.eval({'input',gpuArray(single(bsxfun(@minus,patches,avgImg)))});
    feat=gather(net1.vars(23).value);
    
    feat = mat2cell(feat,size(feat,1),size(feat,2),size(feat,3),ones(1,scale_num));
    
    feat = cellfun(@(fea) imResample(fea, sz_window(1:2)),feat ,'UniformOutput',false);
   
    feat = cellfun(@(fea) bsxfun(@times, fea, cos_window), feat,'UniformOutput',false);

    [hf,wf,cf]=size(feat{1});
    feat = cellfun( @(fea) reshape(reshape(fea,hf*wf,cf)*coeff,hf,wf,num_channels),  feat,'UniformOutput',false );
    
     featPCA = cell2mat(feat);
    
    net_online.eval({'input1',gpuArray(featPCA)});
    regression_map=gather(net_online.vars(net_online.getVarIndex('sum_1')).value);
    
    value = max(reshape(regression_map,[],scale_num));
    
    if value(1)<threshold        
        target_szU=target_sz; 
        scaling=1;
        return;
    end
     
% for i=1:length(scale)
%     target_sz1=round(target_sz*scale(i));
%     window_sz=get_search_window(target_sz1,im_sz);
%     patch = get_subwindow(im, pos, window_sz);    
%     patch1=double(imresize(patch,window_szo));
%        
%     net1.eval({'input',gpuArray(single(bsxfun(@minus,patch1,avgImg)))});
%     feat=gather(net1.vars(23).value);
%     feat = imResample(feat, sz_window(1:2));
%     feat = bsxfun(@times, feat, cos_window);
% 
%     [hf,wf,cf]=size(feat);
%     feat_=reshape(feat,hf*wf,cf);
%     feat_=feat_*coeff;
%     featPCA=reshape(feat_,hf,wf,num_channels);
% %lx    net_online.eval({'input1',gpuArray(featPCA),'input2',gpuArray(featPCA1st)});
%     net_online.eval({'input',gpuArray(featPCA)});
% %lx     regression_map=gather(net_online.vars(10).value);
%     regression_map=gather(net_online.vars(net_online.getVarIndex('sum_1')).value);
%     value(i)=max(regression_map(:));
%     
%     if value(1)<threshold        
%         target_szU=target_sz; 
%         scaling=1;
%         return;
%     end
% end


[~,id]=max(value);%.*[1 1 0.99]');

% if id==1
%     target_szU=target_sz;    
%     scaling=1;
%     return;    
% end

% target_szU=0.4*target_sz+0.6*round(target_sz*scale(id));
scaling = scale(id);
target_szU=round(target_sz*scaling);
end

