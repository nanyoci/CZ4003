%% 1: Point processing

close all
clear all

pc = imread('mrt-train.jpg');
whos pc;
p = rgb2gray(pc);
imshow(p);

min_p = min(p(:))   % min = 13
max_p = max(p(:))   % max = 204

p1(:,:) = imsubtract(p(:,:), double(min_p));
p2(:,:) = immultiply(p1(:,:), double(255/double(max_p-min_p)));

min_p2 = min(p2(:))   % min = 0
max_p2 = max(p2(:))   % max = 255

% imshow(p2);
subplot(1,2,1); imagesc(p); axis off; title('Original Image');
subplot(1,2,2); imagesc(p2); axis off; title('New Enhanced Image');
colormap(gray);

%% 2: Histogram Equalization

imhist(p, 10)

p3 = histeq(p, 255);
p4 = histeq(p3, 255);

size(pc)

% display histograms
subplot(2,2,1); imhist(p, 10); axis on; title('Before Histogram Equalization (10 bins)');
subplot(2,2,2); imhist(p, 256); axis on; title('Before Histogram Equalization (256 bins)');
subplot(2,2,3); imhist(p3, 10); axis on; title('After Histogram Equalization (10 bins)');
subplot(2,2,4); imhist(p3, 256); axis on; title('After Histogram Equalization (256 bins)');

% display images
subplot(2,3,1); imagesc(p); axis off;  title('Original Image'); 
subplot(2,3,2); imagesc(p2); axis off;  title('Contrast Stretching');
subplot(2,3,3); imagesc(p3); axis off;  title('Histogram Equalization');
subplot(2,3,4); imhist(p, 256); axis on; title('Original Histogram');
subplot(2,3,5); imhist(p2, 256); axis on; title('Contrast Stretching Histogram');
subplot(2,3,6); imhist(p3, 256); axis on; title('Histogram Equalization Histogram');
colormap(gray);


%% 3: Spatial filtering
close all
clear all

%% 3.1: Gaussian filtering
x = -2:2;
y = -2:2;

[X,Y] = meshgrid(x,y)
sigma = 1;
h1 = exp(-((X.^2 + Y.^2)/(2*sigma^2)))/(2*pi*sigma^2)
h1_norm = h1./sum(h1(:))

sigma = 2;
h2 = exp(-((X.^2 + Y.^2)/(2*sigma^2)))/(2*pi*sigma^2)
h2_norm = h2./sum(h2(:))

subplot(1,2,1); mesh(h1_norm); axis on; title('σ = 1');
subplot(1,2,2); mesh(h2_norm); axis on; title('σ = 2');

%% 3.1.1: ntu-gn
n = imread('ntu-gn.jpg');
n1 = conv2(n,h1);
n2 = conv2(n,h2);
 
subplot(2,3,1); imagesc(n) ; axis off; title('NTU Gaussian Noise');
subplot(2,3,2); imagesc(n1) ; axis off; title('NTU Gaussian Noise (σ = 1)');
subplot(2,3,3); imagesc(n2) ; axis off; title('NTU Gaussian Noise (σ = 2)');
colormap(gray);

%% 3.1.2: ntu-sp
m = imread('ntu-sp.jpg');
m1 = conv2(m,h1);
m2 = conv2(m,h2);

 
subplot(2,3,4); imagesc(m) ; axis off; title('NTU Speckle Noise');
subplot(2,3,5); imagesc(m1) ; axis off; title('NTU Speckle Noise (σ = 1)');
subplot(2,3,6); imagesc(m2) ; axis off; title('NTU Speckle Noise (σ = 2)');
colormap(gray);


%% 4: Median Filtering

n = imread('ntu-gn.jpg');
n1 = medfilt2(n,[3,3]);
n2 = medfilt2(n,[5,5]);

% display ntu-gn results
subplot(2,3,1); imagesc(n) ; axis off; title('NTU Gaussian Noise');
subplot(2,3,2); imagesc(n1) ; axis off; title('NTU Gaussian Noise (3 x 3)');
subplot(2,3,3); imagesc(n2) ; axis off; title('NTU Gaussian Noise (5 x 5)');
colormap(gray);

m = imread('ntu-sp.jpg');
m1 = medfilt2(m,[3,3]);
m2 = medfilt2(m,[5,5]);

% display ntu-sp results
subplot(2,3,4); imagesc(m) ; axis off; title('NTU Speckle Noise');
subplot(2,3,5); imagesc(m1) ; axis off; title('NTU Speckle Noise (3 x 3)');
subplot(2,3,6); imagesc(m2) ; axis off; title('NTU Speckle Noise (5 x 5)');
colormap(gray);

%% 3.2 Suppressing Noise Interference Patterns
close all
clear all

f = imread('pck-int.jpg');

F = fft2(f);
S = abs(F);
Ss = fftshift(S.^0.1);      %non-linearly scale the power

% imagesc(Ss);colormap('default');
% imagesc(S.^0.1)
subplot(1,2,1); imagesc(S.^0.1) ; axis off; title('Power Spectrum of original image (without fftshift)');
subplot(1,2,2); imagesc(Ss) ; axis off; title('Power Spectrum of original image (with fftshift)');
colormap('default');

% x = 241, y = 9 and x = 17, y= 249
x1 = 241; y1 = 9;
x2 = 17; y2 = 249;

F_new = F;
F_new(x1-2:x1+2, y1-2:y1+2) = 0;
F_new(x2-2:x2+2, y2-2:y2+2) = 0;
S_new = abs(F_new);
Ss_new = fftshift(S_new.^0.1);

% display power spectrum of enhanced image
subplot(1,2,1); imagesc(S_new.^0.1); axis off; title('Power Spectrum of enhanced image (without fftshift)');
subplot(1,2,2); imagesc(Ss_new) ; axis off; title('Power Spectrum of enhanced image (with fftshift)');
colormap('default');

 
f_new = ifft2(F_new);
subplot(1,2,1); imagesc(f) ; axis off; title('Original image');colormap(gray);
subplot(1,2,2); imagesc(f_new) ; axis off; title('Enhanced image');colormap(gray);

%% 3.2.1 Further Enhanced Image

F2_new = F_new;
F2_new(x1-7:x1+7, y1-7:y1+7) = 0;
F2_new(x2-7:x2+7, y2-7:y2+7) = 0;
F2_new(x1-230:x1+15, y1) = 0;
F2_new(x2-15:x2+230, y2) = 0;
F2_new(x1, y1-7:y1+240) = 0;
F2_new(x2, y2-240:y2+7) = 0;
S2_new = abs(F2_new);
S2s_new = fftshift(F2_new);
f2_new = ifft2(F2_new);

% display power spectrum of further enhance image
imagesc(S2_new.^0.1) ; axis off; title('Power Spectrum of further enhanced image (without fftshift)');
colormap('default');

% display the further enhance image
imagesc(f2_new) ; axis off; title('Further Enhanced image');
colormap(gray);

% display orginal, enhanced, further enhanced image
subplot(1,3,1); imagesc(f) ; axis off; title('Original image');
subplot(1,3,2); imagesc(f_new) ; axis off; title('Enhanced image');
subplot(1,3,3); imagesc(f2_new) ; axis off; title('Further Enhanced image');
colormap(gray);

%% 3.2.2 primate-cage
close all
clear all
f_rgb = imread('primate-caged.jpg');
f = rgb2gray(f_rgb);

F = fft2(f);
S = abs(F);
Ss = fftshift(S.^0.1);      %non-linearly scale the power
imagesc(Ss);

x1 = 134; y1 = 119;
x2 = 124; y2 = 139;
x3 = 138; y3 = 108;
x4 = 120; y4 = 150;
x5 = 143; y5 = 97;
x6 = 115; y6 = 161;

Fs_new = fftshift(F);

threshold = 2;
Fs_new(x1-threshold:x1+threshold, y1-threshold:y1+threshold) = 0;
Fs_new(x2-threshold:x2+threshold, y2-threshold:y2+threshold) = 0;
Fs_new(x3-threshold:x3+threshold, y3-threshold:y3+threshold) = 0;
Fs_new(x4-threshold:x4+threshold, y4-threshold:y4+threshold) = 0;
Fs_new(x5-threshold:x5+threshold, y5-threshold:y5+threshold) = 0;
Fs_new(x6-threshold:x6+threshold, y6-threshold:y6+threshold) = 0;
f_new = ifft2(ifftshift(Fs_new));

subplot(1,3,1); imagesc(f) ; axis on; title('Original image');
subplot(1,3,2); imagesc(abs(f_new)) ; axis on; title('Enhanced image');
subplot(1,3,3); imagesc(abs((Fs_new).^0.1)) ; axis on; title('Power Spectrum of Enhanced image (shifted)');
colormap(gray)

%% Undoing Perspective Distortion of Planar Surface
close all
clear all

P = imread('book.jpg');
imagesc(P);
[X, Y] = ginput(4);

% A4 210 x 297
Xim = [0 210 210 0]; 
Yim = [0 0 297 297];

A = [
    [X(1), Y(1), 1, 0, 0, 0, -Xim(1)*X(1), -Xim(1)*Y(1)];
    [0, 0, 0, X(1), Y(1), 1, -Yim(1)*X(1), -Yim(1)*Y(1)];
    [X(2), Y(2), 1, 0, 0, 0, -Xim(2)*X(2), -Xim(2)*Y(2)];
    [0, 0, 0, X(2), Y(2), 1, -Yim(2)*X(2), -Yim(2)*Y(2)];
    [X(3), Y(3), 1, 0, 0, 0, -Xim(3)*X(3), -Xim(3)*Y(3)];
    [0, 0, 0, X(3), Y(3), 1, -Yim(3)*X(3), -Yim(3)*Y(3)];
    [X(4), Y(4), 1, 0, 0, 0, -Xim(4)*X(4), -Xim(4)*Y(4)];
    [0, 0, 0, X(4), Y(4), 1, -Yim(4)*X(4), -Yim(4)*Y(4)];
];
v = [Xim(1); Yim(1); Xim(2); Yim(2); Xim(3); Yim(3); Xim(4); Yim(4);];
u = A\v;
U = reshape([u;1], 3, 3)'           % matrix of m11 to m31

w = U*[X'; Y'; ones(1,4)]           %[[kxim][kyim][k]]  
w = w ./ (ones(3,1) * w(3,:))       %[[xim][yim][1]] 
T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);

imagesc(P2);


