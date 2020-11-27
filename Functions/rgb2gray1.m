function G = rgb2gray1(X,datatype)
% function G = rgb2gray1(X) Convert RGB-like real data to gray-like output.

if size(X,3)==1; G=X; return, end
if nargin<2, datatype = 'double'; end
[M,N,three] = size(X);
X = reshape(X,[M*N,three]);

% Calculate transformation matrix
T = inv([1.0 0.956 0.621; 1.0 -0.272 -0.647; 1.0 -1.106 1.703]);
coef = T(1,:)';

switch datatype
    case 'single'
        G = reshape(single(X)*coef,[M,N]);
    case 'double'
        G = reshape(double(X)*coef,[M,N]);
end