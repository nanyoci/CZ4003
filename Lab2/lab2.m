%% Edge Detection
% download `macritchie.jpg’, convert to grayscale, display the image.
I = imread('macritchie.jpg');
I = rgb2gray(I);
imshow(I);

%%
% create 3x3 horizontal & vertical Sobel masks, 
sobel_h = [
    -1 -2 -1; 
    0 0 0; 
    1 2 1;
];

sobel_v = [
    -1 0 1; 
    -2 0 2; 
    -1 0 1;
];

% filter the image using conv2
f_h = conv2(I, sobel_h);
f_v = conv2(I, sobel_v);

% display the edge-filtered images
subplot(1,2,1); imagesc(uint8(f_h)); axis off; title('Convoluted with horizontal filter');
subplot(1,2,2); imagesc(uint8(f_v)); axis off; title('Convoluted with vertical filter');
colormap(gray);

%%
% combined edge image by squaring and adding the squared images
E = f_h.^2 + f_v.^2;
imshow(uint8(E));

%%
% threshold the edge image E at value t by 
t = 50;
Et = E>t;

% try different threshold values
t2 = 100;
Et2 = E>t2;
t3 = 5000;
Et3 = E>t3;
t4 = 10000;
Et4 = E>t4;
t5 = 50000;
Et5 = E>t5;
t6 = 100000;
Et6 = E>t6;

% display the binary edge images 
subplot(3,2,1); imagesc(uint8(Et)); axis off;
subplot(3,2,2); imagesc(uint8(Et2)); axis off;
subplot(3,2,3); imagesc(uint8(Et3)); axis off;
subplot(3,2,4); imagesc(uint8(Et4)); axis off;
subplot(3,2,5); imagesc(uint8(Et5)); axis off;
subplot(3,2,6); imagesc(uint8(Et6)); axis off;
colormap(gray);

%%
tl=0.04;
th=0.1; 
sigma=1.0;
E = edge(I,'canny',[tl th],sigma);
imagesc(uint8(E));
colormap(gray);
axis off;

% try different values of sigma ranging from 1.0 to 5.0
sigma2=2.0;
E2 = edge(I,'canny',[tl th],sigma2);
sigma3=3.0;
E3 = edge(I,'canny',[tl th],sigma3);
sigma4=5.0;
E4 = edge(I,'canny',[tl th],sigma4);

% display edge images
subplot(2,2,1); imagesc(uint8(E)); axis off;
subplot(2,2,2); imagesc(uint8(E2)); axis off;
subplot(2,2,3); imagesc(uint8(E3)); axis off;
subplot(2,2,4); imagesc(uint8(E4)); axis off;
colormap(gray);

%%
tl2 = 0.01;
tl3 = 0.08;
E5 = edge(I,'canny',[tl2 th],sigma);
E6 = edge(I,'canny',[tl3 th],sigma);

% display edge images
subplot(2,2,1); imagesc(uint8(E5)); axis off;
subplot(2,2,2); imagesc(uint8(E6)); axis off;
colormap(gray);

%%
[H, xp] = radon(E);
subplot(1,1,1); imagesc(0:179, xp, H); colormap(gray);
xlabel('\theta (degrees)')
ylabel('y''')

%% 
maximum = max(max(H));
[d,theta]=find(H==maximum);
radius = xp(d);
theta = theta - 1;

%%
[A, B] = pol2cart(theta*pi/180, radius)
B = -B

C = (A^2 + B^2) + 179*A + 145*B

%%
xl = 0;
yl = (C - A * xl) / B
xr = 357;
yr = (C - A * xr) / B

%%
imagesc(I);
line([xl xr], [yl yr]);

%%
% Experiment 3
l = imread('corridorl.jpg'); 
l = rgb2gray(l);
imagesc(l);

r = imread('corridorr.jpg');
r = rgb2gray(r);
imagesc(r);

disp = imread('corridor_disp.jpg');
imagesc(disp);

%%
D = disparityMap(l, r, 11, 11);
imshow(D,[-15 15]);colormap(gray);

%%
l = imread('triclopsi2l.jpg'); 
l = rgb2gray(l);
% imagesc(l);

r = imread('triclopsi2r.jpg');
r = rgb2gray(r);
% imagesc(r);

disp = imread('triclopsid.jpg');
% imagesc(disp); colormap(gray);

D = disparityMap(l, r, 11, 11);
imshow(D,[-15 15]);colormap(gray);

%%
function map = disparityMap(pl, pr, template_x, template_y)
    [m, n] = size(pl);
    map = ones(m - template_y + 1 , n - template_x + 1);
    x_half = floor(template_x/2);
    y_half = floor(template_y/2);
    for i = 1+x_half:n-x_half
        for j = 1+y_half:m-y_half
            patch_l = pl(j-y_half:j+y_half,i-x_half:i+x_half);
            lower_x = max(1+x_half,i-15);
            upper_x = min(n-x_half,i+15);
            min_ssd = inf; 
            min_ssd_idx = lower_x;
            for x = lower_x:upper_x
                patch_r = pr(j-y_half:j+y_half,x-x_half:x+x_half);
                temp_r = rot90(patch_r,2);
                i_square = conv2(patch_r,temp_r, 'same');
                c = 2.*conv2(patch_l,temp_r, 'same');
                ssd = i_square(y_half+1,x_half+1) - c(y_half+1,x_half+1);
                if ssd < min_ssd
                    min_ssd = ssd;
                    min_ssd_idx = x;
                end
            end
            map(j-y_half, i-x_half) = i - min_ssd_idx;
        end
    end
end


