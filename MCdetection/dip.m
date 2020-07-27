function y = dip(ImgName)
%Img = uint8(mat2gray(imread(ImgName))*255);
%Img = adapthisteq(Img);
%Img=histeq(Img);
%Img=imadjust(Img)
%Img=wiener2(imadjust(Img),[3,3]);
Img=imread(ImgName);
s = size(Img);
if length(s)==3
    Img = rgb2gray(imread(ImgName));
end

ns = [128*ceil(s(1)/128),128*ceil(s(2)/128)];
Img_n(1:ns(1),1:ns(2)) = Img;
for i=0:(ns(1)/128)-1
    for j=0:(ns(2)/128)-1
        P = Img_n(i*128+1:i*128+128,j*128+1:j*128+128); 
        trans = mgramfilt(im2uint8(P));
        New_Img(i*128+1:i*128+128,j*128+1:j*128+128) = trans; 
    end
end
y=(im2uint8(New_Img));
