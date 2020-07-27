function y = trans(binary_array)
s = size(binary_array);

ns = [128*ceil(s(1)/128),128*ceil(s(2)/128)];
Img_n(1:ns(1),1:ns(2)) = binary_array;
for i=0:(ns(1)/128)-1
    for j=0:(ns(2)/128)-1
        P = Img_n(i*128+1:i*128+128,j*128+1:j*128+128); 
        trans = mgramfilt(im2uint8(P));
        New_Img(i*128+1:i*128+128,j*128+1:j*128+128) = trans; 
    end
end
y=(im2uint8(New_Img));