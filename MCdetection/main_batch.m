clear;clc;
D = '.\testingroi';
SV='.\testingtopo';
S = dir(fullfile(D,'*.bmp')); % pattern to match filenames.
for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    MC_bin=dip(F);
    MC_p =im2uint8(bwareaopen(MC_bin, 5));  % remove all objects whose area less than 5
    imwrite(MC_p,fullfile(SV,S(k).name))
    fprintf('Sub folder #%d = %s\n', k, S(k).name);
end