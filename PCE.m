function [Out,Out0] = PCE(C,shift_range,squaresize)

% -------------------------------------------------------------------------
% Copyright (c) 2012 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Contact: mgoljan@binghamton.edu | January 2012,  Last modified Oct 2018 
%          http://dde.binghamton.edu/download/camera_fingerprint
% -------------------------------------------------------------------------
% Peak-to-correlation energy obtained from correlation surface restricted to possible shifts due to cropping
% In this implementation of Peak-to-correlation energy PCE carries the sign of the peak (can be negative)
% Input:   
%    C               cross-correlation surface calculated by crosscorr.m
%    shift_range     maximum shift is from [0,0] to [shift_range]
%    squaresize      remove the peak neighborhood of size (squaresize x squaresize)
% Output: 
%    Out.PCE         peak-to-correlation energy (PCE)
%    Out.PeakLocation  location of the primary peak, [0 0] when correlated signals are not shifted to each other
%    Out.pvalue      probability of obtaining peakheight or higher (under Gaussian assumption)
%    Out.P_FA        probability of false alarm (increases with increasing range of admissible shifts (shift_range)   
%    Out0 contains results for the test under assumption of no cropping (equivalent to Out0 = PCE(C))
% Examples:
%  Measure correlation of Noise1 and Noise2 in terms of PCE:
%  [pce1,pce0] = PCE(crosscorr(Noise1,Noise2),size(Noise)-1);   % search all possible shifts
%  C = crosscorr(Noise1,Noise2); Out0 = PCE(C)      % Out0.PCE = Out.PCE and Out.P_FA = Out.pvalue when no shifts are considered

if nargin<3, squaresize = 11; end               % default neighborhood of the peak
if nargin<2, shift_range = [0,0]; end           % default: no shifts (no cropping) considered 
if any(shift_range>=size(C)), 
    shift_range = min(shift_range,size(C)-1);   % all possible shift in at least one dimension 
end 

if C==0;            % the case when cross-correlation C has zero energy (see crosscor2)
    Out.PCE = 0; 
    Out.pvalue = 1; 
    Out.PeakLocation = [0,0];
    return, 
end  

Cinrange = C(end-shift_range(1):end,end-shift_range(2):end);  	% C(end,end) location corresponds to no shift of the first matrix argument of crosscor2.m
% disp(size(Cinrange))
[max_cc, imax] = max(Cinrange(:));
[ypeak, xpeak] = ind2sub(size(Cinrange),imax(1));
Out.peakheight = Cinrange(ypeak,xpeak);
clear Cinrange
Out.PeakLocation = shift_range+[1,1]-[ypeak, xpeak];

C_without_peak = RemoveNeighborhood(C,size(C)-Out.PeakLocation,squaresize);
correl = C(end,end);        clear C

% signed PCE, peak-to-correlation energy    (the sign added as of 10/26/2011)
PCE_energy = mean(C_without_peak.*C_without_peak);
Out.PCE = Out.peakheight.^2/PCE_energy * sign(Out.peakheight);

% p-value
% [mu, alpha, beta] = GGDfit(C_without_peak);                   % Generalized Gaussian fit
% Out.pvalue = 1- GGcdf(Out.peakheight,mu,alpha,beta);
Out.pvalue = 1/2*erfc(Out.peakheight/sqrt(PCE_energy)/sqrt(2));     % under simplifying assumption that C are samples from Gaussian pdf 
[Out.P_FA,Out.log10P_FA] = FAfromPCE(Out.PCE,prod(shift_range+1));
% pvalue = 1/2*erfc(sqrt(PCE/2));  => PCE = 2*(erfcinv(2*pvalue))^2;

Out0.PCE = correl^2/PCE_energy;
% Out0.pvalue = 1/2*erfc(Out0.PCE^2/sqrt(2));  
[Out0.P_FA,Out0.log10P_FA] = FAfromPCE(Out0.PCE,1);
end

%----------------------------------------
function Y = RemoveNeighborhood(X,x,ssize)
% Remove a 2-D neighborhood around x=[x1,x2] from matrix X and output a 1-D vector Y
% ssize     square neighborhood has size (ssize x ssize) square
[M,N] = size(X);
radius = (ssize-1)/2;
X = circshift(X,[radius-x(1)+1,radius-x(2)+1]);
Y = X(ssize+1:end,1:ssize);   Y = Y(:);
Y = [Y;X(M*ssize+1:end)'];
% disp(size(Y)) 65415 = 256*256 - 121
end

function [FA,log10FA] = FAfromPCE(pce,search_space)
% Calculates false alarm probability from having peak-to-cross-correlation (PCE) measure of the peak
% pce           PCE measure obtained from PCE.m
% seach_space   number of correlation samples from which the maximum is taken
%  USAGE:   FA = FAfromPCE(31.5,32*32);

% p = 1/2*erfc(sqrt(pce)/sqrt(2));
[p,logp] = Qfunction(sign(pce)*sqrt(abs(pce)));
if pce<50, 
    FA = 1-(1-p).^search_space;
else
    FA = search_space*p;                % an approximation
end

if FA==0,
    FA = search_space*p;   
    log10FA = log10(search_space)+logp*log10(exp(1));
else 
    log10FA = log10(FA);
end    
end

function [Q,logQ] = Qfunction(x)
% Calculates probability of a Gaussian variable N(0,1) taking value larger than x

if x<37.5, 
    Q = 1/2*erfc(x/sqrt(2));
    logQ = log(Q);
else
    Q = 1./sqrt(2*pi)./x.*exp(-(x.^2)/2);
    logQ = -(x.^2)/2 - log(x)-1/2*log(2*pi);
end
end