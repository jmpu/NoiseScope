function Noise = NoiseExtractFromImage(image,sigma,color,noZM, residual_dir, confidence)
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
% Contact: mgoljan@binghamton.edu | January 2012
%          http://dde.binghamton.edu/download/camera_fingerprint
% -------------------------------------------------------------------------
% function Noise = NoiseExtractFromImage(image,sigma,color,noZM) estimates PRNU from one image
% INPUT:
%   image      test image filename
%   sigma      std of noise to be used for identicication (recomended value between 2 and 3)
%   color      if color then output Noise will be a colored noise
%   noZM       if noZM then the output noise residual is not processed by zero-mean filter
% OUTPUT:
%   Noise      extracted noise from the input image, a rough estimate of PRNU fingerprint
% EXAMPLE:     extract grayscale noise residual (with local std = 2) from a JPEG image 
% Noise = NoiseExtractFromImage('DSC00123.JPG',2);
% -------------------------------------------------------------------------
% [1] M. Goljan, T. Filler, and J. Fridrich. Large Scale Test of Sensor 
% Fingerprint Camera Identification. In N.D. Memon and E.J. Delp and P.W. Wong and 
% J. Dittmann, editors, Proc. of SPIE, Electronic Imaging, Media Forensics and 
% Security XI, volume 7254, pages % 0I�01�0I�12, January 2009.
% -------------------------------------------------------------------------

%%% ----- Parameters ----- %%%
L=4;            % number of wavelet decomposition levels (between 2-5 as well)

addpath(strcat(pwd,'/Functions'))
addpath(strcat(pwd,'/Filter'))

if nargin<4, noZM=0;  end
if nargin<3, color=0; end

if ischar(image),
    X = imread(image);
else X = image;
    clear image,
end

[M0,N0,three]=size(X);
    datatype = class(X);
    switch datatype,                    % convert to [0,255]
        case 'uint8',  X = double(X);
        case 'uint16', X = double(X)/65535*255;
    end

qmf = MakeONFilter('Daubechies',8);

if three~=3,
    Noise = NoiseExtract(X,qmf,sigma,L);
else
    Noise = zeros(size(X));
    for j=1:3
        Noise(:,:,j) = NoiseExtract(X(:,:,j),qmf,sigma,L);
    end
    if ~color
        Noise = rgb2gray1(Noise);
    end
end
if noZM
    'not removing the linear pattern';
else
    Noise = ZeroMeanTotal(Noise);
end
Noise = single(Noise);
[filepath,name,ext] = fileparts(image);
if nargin == 5
    save(strcat(residual_dir, name, '.mat'), 'Noise');
    %imwrite(Noise, strcat(residual_dir, name, ext));
end
