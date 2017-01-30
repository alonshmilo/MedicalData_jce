close all
clear all
my_ui = uipanel;
ax1 = axes(      'Parent'                , my_ui, ...
                'Units'                  , 'pixels', ...
                'Position'              , [100 100 1000 1000]);
imagesc1 = imagesc(zeros(1, 'uint8'), ...
                'Parent'                , ax1, ...
                'CDataMapping'          , 'scaled');
ax2 = axes(      'Parent'                , my_ui, ...
                'Units'                  , 'pixels', ...
                'Position'              , [100 100 1000 1000]);
imagesc2= imagesc(zeros(1, 'uint8'), ... 
                'Parent'                , ax2, ...
                'CDataMapping'          , 'scaled');            
            

E = imread('peppercorn_hill2.png');
E = rgb2gray(E);

I = imread('peppercorn_hill_influence_map2.png');
I = rgb2gray(I);

set(imagesc1,'CData', E); 
set(imagesc2,'CData', I); 
set(imagesc2,'AlphaData', 0.2); 
set(imagesc1,'AlphaData', 0.2);  