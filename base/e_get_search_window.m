function window_sz = e_get_search_window( target_sz, im_sz)
% GET_SEARCH_WINDOW

% if(target_sz(1)/target_sz(2) > 2)
%     % For objects with large height, we restrict the search window with padding.height
%     window_sz = floor(target_sz.*[1+padding.height, 1+padding.generic]);
%     
% elseif(prod(target_sz)/prod(im_sz(1:2)) > 0.05)
%     % For objects with large height and width and accounting for at least 10 percent of the whole image,
%     % we only search 2x height and width
%     window_sz=floor(target_sz*(1+padding.large));
%     
% else
%     %otherwise, we use the padding configuration
%     window_sz = floor(target_sz * (1 + padding.generic));

sw_scale = 5; %5

ratio=target_sz(:,1)./target_sz(:,2);
window_sz = zeros(size(target_sz));

for i=1:size(target_sz,1)
    if ratio(i)>1    
        window_sz(i,:)=round(target_sz(i,:).*[sw_scale,sw_scale*ratio(i)]);
    else
        window_sz(i,:)=round(target_sz(i,:).*[sw_scale/ratio(i),sw_scale]);
    end
end

%window_sz=round(target_sz.*[5,9]);
window_sz=window_sz-mod(window_sz,2)+1;

end

