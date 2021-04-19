function [net_online,target_sz] = initNet_base_4diconv(target_sz1)
%Init network

 global num_channels;
% %channel=64;
 global flag;
 flag=0;


channel=64;

rw=ceil(target_sz1(2)/2);
rh=ceil(target_sz1(1)/2);
fw=2*rw+1;
fh=2*rh+1;

target_sz = [fw,fh];

net_online=dagnn.DagNN();

net_online.addLayer('conv11', dagnn.Conv('size', [fw,fh,num_channels,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1]), 'input1', 'conv_11', {'conv11_f', 'conv11_b'});

f = net_online.getParamIndex('conv11_f') ;
net_online.params(f).value=single(randn(fh,fw,num_channels,1) /...
    sqrt(rh*rw*num_channels))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv11_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1e3;

%DiConv  mask1 
mask1=zeros(fh,fw);%产生fh*fw维的全1矩阵
remainder1 = fh/4;
remainder2 = fw/4;
mask1(1:fh/4,1:fw/4)=1;

mask2=zeros(fh,fw);
mask2(fh/4+1:fh/2,fw/4+1:fw/2)=1;

mask3=zeros(fh,fw);
mask3(fh/2+1:3*(fh/4),fw/2+1:3*(fw/4))=1;

mask4=zeros(fh,fw);
mask4(3*(fh/4)+1:end,3*(fw/4)+1:end)=1;

%Diconv1
net_online.addLayer('conv12', DiConv('size', [fw,fh,num_channels,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1],'mask',mask1), 'input1', 'conv_12', {'conv12_f', 'conv12_b'});

f = net_online.getParamIndex('conv12_f') ;
net_online.params(f).value=single(randn(fh,fw,num_channels,1) /...
    sqrt(rh*rw*num_channels*1))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv12_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1e3;

%Diconv2
net_online.addLayer('conv13', DiConv('size', [fw,fh,num_channels,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1],'mask',mask2), 'input1', 'conv_13', {'conv13_f', 'conv13_b'});

f = net_online.getParamIndex('conv13_f') ;
net_online.params(f).value=single(randn(fh,fw,num_channels,1) /...
    sqrt(rh*rw*num_channels*1))/1e8;
net_online.params(f).learningRate=10;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv13_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=20;
net_online.params(f).weightDecay=1e3;

%Diconv3
net_online.addLayer('conv14', DiConv('size', [fw,fh,num_channels,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1],'mask',mask3), 'input1', 'conv_14', {'conv14_f', 'conv14_b'});

f = net_online.getParamIndex('conv14_f') ;
net_online.params(f).value=single(randn(fh,fw,num_channels,1) /...
    sqrt(rh*rw*num_channels*1))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv14_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1e3;

%Diconv4
net_online.addLayer('conv15', DiConv('size', [fw,fh,num_channels,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1],'mask',mask4), 'input1', 'conv_15', {'conv15_f', 'conv15_b'});

f = net_online.getParamIndex('conv15_f') ;
net_online.params(f).value=single(randn(fh,fw,num_channels,1) /...
    sqrt(rh*rw*num_channels*1))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv15_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1e3;


% Res Learning 

% net_online.addLayer('hinge_loss',...
%     dagnn.Loss('loss', 'hinge'),{'sum_1','label_binary'},'objective_h');
net_online.addLayer('sum1', WeightSum1(),{'conv_12','conv_13','conv_14','conv_15','conv_11'},'sum_1');

net_online.addLayer('L2Loss',...
    RegressionL2Loss(),{'sum_1','label'},'objective');

end



