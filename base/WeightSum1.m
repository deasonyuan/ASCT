classdef WeightSum1 < dagnn.ElementWise

  properties (Transient)
    numInputs
	beta1
    beta2
    beta3
    beta4
	
  end

  methods
    function outputs = forward(obj, inputs, params)
      global pre_sz1;
      global flag;
      if flag==0
          [vert, horiz] = find(inputs{5}== max(max(inputs{5})),1);
          [h,w]=size(inputs{1});
          pre_sz1= [vert,horiz,h,w];
          flag=1;
      end  
      obj.numInputs = numel(inputs) ;
	 
      [vert_1, horiz_1] = find(inputs{1}== max(inputs{1}(:)),1);
      max_1 = max(inputs{1}(:));
      mean_1 = mean(inputs{1}(:));
      std_1 = std(inputs{1}(:));
      psk1 = (max_1-mean_1)/std_1;
      
      [vert_2, horiz_2] = find(inputs{2}== max(inputs{2}(:)),1);
      max_2 = max(inputs{2}(:));
      mean_2 = mean(inputs{2}(:));
      std_2 = std(inputs{2}(:));
      psk2 = (max_2-mean_2)/std_2;
      
      [vert_3, horiz_3] = find(inputs{3}== max(inputs{3}(:)),1);
      max_3 = max(inputs{3}(:));
      mean_3 = mean(inputs{3}(:));
      std_3 = std(inputs{3}(:));
      psk3 = (max_3-mean_3)/std_3;
      
      [vert_4, horiz_4] = find(inputs{4}== max(inputs{4}(:)),1);
      max_4 = max(inputs{4}(:));
      mean_4 = mean(inputs{4}(:));
      std_4 = std(inputs{4}(:));
      psk4 = (max_4-mean_4)/std_4;
      
      if abs(vert_1-ceil(pre_sz1(1)/2))>ceil(pre_sz1(3)/4)&&abs(horiz_1-ceil(pre_sz1(2)/2))>ceil(pre_sz1(4)/4)
         bbeta1=psk1*0.75;
        
      else
          bbeta1 = psk1;
      end
      if abs(vert_2-ceil(pre_sz1(1)/2))>ceil(pre_sz1(3)/4)&&abs(horiz_2-ceil(pre_sz1(2)/2))>ceil(pre_sz1(4)/4)
         bbeta2=psk2*0.75;
      else
          bbeta2 = psk2;
      end
      if abs(vert_3-ceil(pre_sz1(1)/2))>ceil(pre_sz1(3)/4)&&abs(horiz_3-ceil(pre_sz1(2)/2))>ceil(pre_sz1(4)/4)
         bbeta3=psk3*0.75;
      else
          bbeta3 = psk3;
      end
      if abs(vert_4-ceil(pre_sz1(1)/2))>ceil(pre_sz1(3)/4)&&abs(horiz_4-ceil(pre_sz1(2)/2))>ceil(pre_sz1(4)/4)
         bbeta4=psk4*0.75;
      else
          bbeta4 = psk4;
      end

	  obj.beta1 = bbeta1/(bbeta1+bbeta2+bbeta3+bbeta4);
      obj.beta2 = bbeta2/(bbeta1+bbeta2+bbeta3+bbeta4);
      obj.beta3 = bbeta3/(bbeta1+bbeta2+bbeta3+bbeta4);
      obj.beta4 = bbeta4/(bbeta1+bbeta2+bbeta3+bbeta4);
  
      
	if obj.numInputs==5
          outputs{1} = inputs{5} ;
          outputs{1} = outputs{1}+(obj.beta1*inputs{1}+obj.beta2*inputs{2}+obj.beta3*inputs{3}+obj.beta4*inputs{4});
          [vert, horiz] = find(outputs{1}== max(max(outputs{1})),1);
          [h,w]=size(outputs{1});
          pre_sz1= [vert,horiz,h,w];
          
      end

    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)

      if obj.numInputs==5
          derInputs{1} = obj.beta1*derOutputs{1};
		  derInputs{2} = obj.beta2*derOutputs{1} ;
		  derInputs{3} = obj.beta3*derOutputs{1} ;
		  derInputs{4} = obj.beta4*derOutputs{1} ;
                  derInputs{5} = derOutputs{1};
  
      end
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Sum layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = WeightSum(varargin)
      obj.load(varargin) ;
    end
  end
end
