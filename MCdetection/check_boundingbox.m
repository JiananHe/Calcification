clear;clc;
%% image show pair-wise
roi_path='./roi';
mask_path='./topo';
roi_name = dir(fullfile(roi_path,'*.bmp')); 
for k = 1:numel(roi_name)
    %roi=uint8(mat2gray(imread(fullfile(roi_path,roi_name(k).name)))*255);
    roi = imread(fullfile(roi_path,roi_name(k).name));
    if length(size(roi))==3
        roi = rgb2gray(roi);
    end
    mask=imread(fullfile(mask_path,roi_name(k).name))/1.5;
    fig=figure('Name',roi_name(k).name,'NumberTitle','off');
    imshow(roi, 'InitialMag', 'fit')
    hold on
    shade = cat(3, ones(size(roi))/255*255, ones(size(roi))/255*97, ones(size(roi))*0);
    h = imshow(shade);
    hold off
    set(h, 'AlphaData', mask)   
    next = waitforbuttonpress;
    if next==0
        close(fig);
        continue
    else
        break
    end    
end

%% Bounding Box 
% figure
% img = dicomread('DDSM/data/full/mal_ge_four/02460RMLO.dcm');
% mask = imread('DDSM/data/mask/mal_ge_four/mask_02460RMLO.png');
% imshowpair(img,mask,'blend');
% rectangle('position',[977, 1639, 1041, 1597]);
% hold on
% x=1553.779879885854; 
% y=2400.36670033871;
% plot(x, y, 'r*', 'LineWidth', 2, 'MarkerSize', 15);
%% Voxel Num
%roi_pixel_num=find(mask==255);
