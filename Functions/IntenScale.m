function out = IntenScale(in)
% in   are pixel intensities 0<=in<=1
% out  0<out<255

T = 252;
v = 6;
out = exp(-1*(in-T).^2/v);

out(in<T) = in(in<T)/T;
%return;
% DC  = 30;
% out(in<30) = 0.1;