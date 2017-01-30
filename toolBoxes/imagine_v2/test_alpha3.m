
close all
clear all
clc



E = imread('peppercorn_hill2.png');
E = rgb2gray(E);

imshow(E, 'InitialMag', 'fit')

%%
% The bright blob at the upper left is Peppercorn Hill, and the
% flat, dark plateau in the upper middle is North Pond.
%
% Below is an "influence map." This is a visualization of down-hill water
% flow, starting from the peak of Peppercorn Hill.

I = imread('peppercorn_hill_influence_map2.png');
I = rgb2gray(I);

%imshow(I, 'InitialMag', 'fit')

%%
% It's difficult to interpret the influence map image on its own,
% apart from the original DEM.  Let's visualize the two images
% together as follows:
%
% # Display the original DEM image.
% # Display a solid green "image" on top of the original image.
% # Use the influence map pixels to control the transparency of
%   each pixel of the green image.

%imshow(E, 'InitialMag', 'fit')
if (0)

% Make a truecolor all-green image.
green = cat(3, zeros(size(E)), ones(size(E)), zeros(size(E)));
hold on
h = imshow(green);
hold off

%%

% Use our influence map image as the AlphaData for the solid
% green image.
    set(h, 'AlphaData', I/2)
end

%%
% Now it's easy to understand the water flow in the context of the original DEM
% image. We can see that the water flows from the peak into the pond, then out
% the southern end of the pond.

%%
% So there you go, better late than never.  Two more image visualization
% techniques to add to your bag of tricks.

%%
% _Copyright 2009 The MathWorks, Inc._