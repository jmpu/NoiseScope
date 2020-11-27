function X = imcropmiddle(X,sizeout,preference)
% function X = imcropmiddle(X,sizeout,preference) crops the middle portion of a given size
% X         image matrix
% sizeout   size of the output image

if nargin<3, preference='SE'; end
if length(sizeout)>2, sizeout = sizeout(1:2); end

[M,N,three]=size(X);
sizeout = min([[M,N];sizeout]);
switch preference           % the cropped region is off center by 1/2 pixel
    case 'NW'
        M0 = floor((M-sizeout(1))/2);
        M1 = M0+sizeout(1);
        N0 = floor((N-sizeout(2))/2);
        N1 = N0+sizeout(2);
    case 'SW'
        M0 = ceil((M-sizeout(1))/2);
        M1 = M0+sizeout(1);
        N0 = floor((N-sizeout(2))/2);
        N1 = N0+sizeout(2);
    case 'NE'
        M0 = floor((M-sizeout(1))/2);
        M1 = M0+sizeout(1);
        N0 = ceil((N-sizeout(2))/2);
        N1 = N0+sizeout(2);
    case 'SE'
        M0 = ceil((M-sizeout(1))/2);
        M1 = M0+sizeout(1);
        N0 = ceil((N-sizeout(2))/2);
        N1 = N0+sizeout(2);
end        
X = X(M0+1:M1,N0+1:N1,:);
