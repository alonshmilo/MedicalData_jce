%MATLAB
%With this tool, we will create synthetic data for unit test

%Important: Use with the suitable functions from nii toolbox: save_nii and
%make_nii

%Using Phantom in both usages:
%P = phantom(def, n)
%'Shepp-Logan' ? Test image used widely by researchers in tomography
%'Modified Shepp-Logan' (default) ? Variant of the Shepp-Logan phantom in which the contrast is improved for better visual perception

%P = phantom(E, n)
%E = [A,a,b,x0,y0,phi]
%A - Additive intensity value of the ellipse
%a - Length of the horizontal semiaxis of the ellipse
%b - Length of the vertical semiaxis of the ellipse
%x0 - x-coordinate of the center of the ellipse
%y0 - y-coordinate of the center of the ellipse
%phi - Angle (in degrees) between the horizontal semiaxis of the ellipse and the x-axis of the image

%Creation of scans:
P = phantom('Modified Shepp-Logan',240);
nii_result = make_nii(P,[240 240 155],[0 0 0],16,'');
save_nii(nii_result,'scan.nii.gz',0)

%GT:
P1 = phantom([1,0.5,0.5,0.2,0.2,90],256)
nii_result = make_nii(P1,[240 240 155],[0 0 0],16,'')
save_nii(nii_result,'gt.nii.gz',0)

%ROI:
P2 = phantom('Shepp-Logan',256);
nii_result = make_nii(P2,[240 240 155],[0 0 0],16,'')
save_nii(nii_result,'roi.nii.gz',0)

