function [Q,logQ]=Qfunction(x)
% function [Q,logQ]=Qfunction(x)  calculates probability that Gaussian variable N(0,1) takes value larger than x

if x<37.5, 
    Q = 1/2*erfc(x/sqrt(2));
    logQ = log(Q);
else
    Q = 1./sqrt(2*pi)./x.*exp(-(x.^2)/2);
    logQ = -(x.^2)/2 - log(x)-1/2*log(2*pi);
end