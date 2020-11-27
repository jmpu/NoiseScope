function F = gan_img_detection_fea(IMG)
%GAN_IMG_DETECTION_FEA Extract features for detecting GAN generated images.
%
% Input: 
% - IMG: An image array or a path to the image file.
%
% Output:
% - F: The 588-D feature for GAN generated image detection.
%
% The algorithm is proposed in the following paper:
% @article{DBLP:journals/corr/abs-1808-07276,
%   author    = {Haodong Li and Bin Li and Shunquan Tan and Jiwu Huang},
%   title     = {Detection of Deep Network Generated Images Using 
%                Disparities in Color Components},
%   journal   = {CoRR},
%   volume    = {abs/1808.07276},
%   year      = {2018}
% }

if ischar(IMG)
    IMG = imread(IMG);
end

HSV = double(im2uint8(rgb2hsv(IMG)));
YCC = double(rgb2ycbcr(IMG));
IMG = double(IMG);

F = cat(2,...
    matrix_cooc_3D(IMG,true),...
    matrix_cooc_2D(HSV(:,:,1),true),...
    matrix_cooc_2D(HSV(:,:,2),true),...
    matrix_cooc_2D(YCC(:,:,2),true),...
    matrix_cooc_2D(YCC(:,:,3),true));



function F = matrix_cooc_2D(IMG,symm)
I_x  = IMG(2:end-1,2:end-1);
I_n = {IMG(2:end-1,3:end);
    IMG(2:end-1,1:end-2);
    IMG(3:end,  2:end-1);
    IMG(1:end-2,2:end-1)};

clips = [-inf -2;
    -1 -1;
    0 0;
    1 1;
    2 inf]; 
N = size(clips,1);
P = zeros(N,N,N);
for j = 1:length(I_n)
    D = I_x - I_n{j};
    D_ = zeros(size(D));
    for i = 1:N
        D_(D>=clips(i,1)  & D<=clips(i,2)) = i;
    end
    
    D_1 = D_(:,1:end-2);
    D_2 = D_(:,2:end-1);
    D_3 = D_(:,3:end);
    M = zeros(N,N,N);
    for k = 1:numel(D_1)
        if symm && D_1(k)>D_3(k)
            M(D_3(k),D_2(k),D_1(k)) = M(D_3(k),D_2(k),D_1(k))+1;
        else
            M(D_1(k),D_2(k),D_3(k)) = M(D_1(k),D_2(k),D_3(k))+1;
        end
    end
    P = P+M./numel(D_1);
    
    D_1 = D_(1:end-2,:);
    D_2 = D_(2:end-1,:);
    D_3 = D_(3:end,:);
    M = zeros(N,N,N);
    for k = 1:numel(D_1)
        if symm && D_1(k)>D_3(k)
            M(D_3(k),D_2(k),D_1(k)) = M(D_3(k),D_2(k),D_1(k))+1;
        else
            M(D_1(k),D_2(k),D_3(k)) = M(D_1(k),D_2(k),D_3(k))+1;
        end
    end
    P = P+M./numel(D_1);
end

if symm
    P_ = false(N,N,N);
    for i = 1:N
        for j = 1:N
            for k = 1:N
                if i <= k
                    P_(i,j,k) = 1;
                end
            end
        end
    end
    F = P(P_)';
else
    F = P(:)';
end

function F = matrix_cooc_3D(IMG,symm)
I_x  = IMG(2:end-1,2:end-1,:);
I_n = {IMG(2:end-1,3:end  ,:);
    IMG(2:end-1,1:end-2,:);
    IMG(3:end,  2:end-1,:);
    IMG(1:end-2,2:end-1,:)};

base = 2;
N = base^3;
P = zeros(N,N,N);
for j = 1:length(I_n)
    D = I_x > I_n{j};
    D_= D(:,:,1)*base^0+D(:,:,2)*base^1+D(:,:,3)*base^2+1;
    
    D_1 = D_(:,1:end-2);
    D_2 = D_(:,2:end-1);
    D_3 = D_(:,3:end);
    M = zeros(N,N,N);
    for k = 1:numel(D_1)
        if symm && D_1(k)>D_3(k)
            M(D_3(k),D_2(k),D_1(k)) = M(D_3(k),D_2(k),D_1(k))+1;
        else
            M(D_1(k),D_2(k),D_3(k)) = M(D_1(k),D_2(k),D_3(k))+1;
        end
    end
    P = P+M./numel(D_1);
    
    D_1 = D_(1:end-2,:);
    D_2 = D_(2:end-1,:);
    D_3 = D_(3:end,:);
    M = zeros(N,N,N);
    for k = 1:numel(D_1)
        if symm && D_1(k)>D_3(k)
            M(D_3(k),D_2(k),D_1(k)) = M(D_3(k),D_2(k),D_1(k))+1;
        else
            M(D_1(k),D_2(k),D_3(k)) = M(D_1(k),D_2(k),D_3(k))+1;
        end
    end
    P = P+M./numel(D_1);
end

if symm
    P_ = false(N,N,N);
    for i = 1:N
        for j = 1:N
            for k = 1:N
                if i <= k
                    P_(i,j,k) = 1;
                end
            end
        end
    end
    F = P(P_)';
else
    F = P(:)';
end