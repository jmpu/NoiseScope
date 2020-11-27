function [LP,D]=LinearPattern(X)
% function Y=LinearPattern(X,type) output column and row means from all 4 subsignals, subsampling by 2. 
% X     2-D matrix
% LP.r11, LP.r12, LP.r21, LP.r22  are row means (column vector)
% LP.c11, LP.c12, LP.c21, LP.c22  are column means (row vector)
% D     The difference between X and ZeroMean(X) 
%       X-D is the zero-meaned version of X

[M,N] = size(X);
me = mean2(X); 
X = X-me;

LP.r11 = mean(X(1:2:end,1:2:end)')';
LP.c11 = mean(X(1:2:end,1:2:end));
  cm11 = mean2(X(1:2:end,1:2:end));
LP.r12 = mean(X(1:2:end,2:2:end)')';
LP.c12 = mean(X(1:2:end,2:2:end));
   cm12 = mean2(X(1:2:end,2:2:end)); % = -cm  Assuming mean2(X)==0
LP.r21 = mean(X(2:2:end,1:2:end)')';
LP.c21 = mean(X(2:2:end,1:2:end));
   cm21 = mean2(X(2:2:end,1:2:end)); % = -cm  Assuming mean2(X)==0
LP.r22 = mean(X(2:2:end,2:2:end)')';
LP.c22 = mean(X(2:2:end,2:2:end));
   cm22 = mean2(X(2:2:end,2:2:end)); % = cm   Assuming mean2(X)==0
LP.me = me;
LP.cm = [cm11,cm12,cm21,cm22];

clear X
D = zeros(M,N);
[aa,bb] = meshgrid(LP.c11,LP.r11);
D(1:2:end,1:2:end) = aa+bb+me-cm11;
[aa,bb] = meshgrid(LP.c12,LP.r12);
D(1:2:end,2:2:end) = aa+bb+me-cm12;
[aa,bb] = meshgrid(LP.c21,LP.r21);
D(2:2:end,1:2:end) = aa+bb+me-cm21;
[aa,bb] = meshgrid(LP.c22,LP.r22);
D(2:2:end,2:2:end) = aa+bb+me-cm22;
