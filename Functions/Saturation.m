function SaturMap = Saturation(X,gray)
% Determines saturated pixels as those having a peak value (must be over
% 250) and a neighboring pixel of equal value
% X    grayscale or color image with pixels in range 0-255
% SaturMap  binary matrix, 0 - saturated pixels 
% gray      in case of 'gray' saturated pixels in SaturMap (denoted as zeros)
%           are those saturated in at least 2 color channels

if nargin<2, gray=''; end
X = double(X);

M = size(X,1);  N = size(X,2);
SaturMap = ones(size(X)); 

Xh = (X - circshift(X,[0,1]));
Xv = (X - circshift(X,[1,0]));
Satur = Xh & Xv & circshift(Xh,[0,-1]) & circshift(Xv,[-1,0]);

if size(X,3)==3, 
    for j=1:3
        maxX(j) = max(max(X(:,:,j)));
        if maxX(j)>250; 
            SaturMap(:,:,j) = ~((X(:,:,j)==maxX(j)) & ~Satur(:,:,j));
        end
    end
elseif size(X,3)==1,
    maxX = max(max(X)); 
    if maxX>250; 
        SaturMap = ~((X==maxX) & ~SaturMap);    
    end
else error('Invalid matrix dimensions)');
end

switch gray
    case 'gray'
        if size(X,3)==3,
            SaturMap = SaturMap(:,:,1)+SaturMap(:,:,2)+SaturMap(:,:,3);
           SaturMap = (SaturMap>=2);
        end
    otherwise 'do nothing';
end