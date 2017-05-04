% This method converts image to a desired dimentions

function resizeCt = resize(filepath,newName,dimx,dimy,dimz)

CtImageInRegularSize = load_untouch_nii_gzip (filepath);
volumeNiiImage = CtImageInRegularSize.img > 0;
figure;imshow(volumeNiiImage(:,:,3));
resizeCt = CTresize2((volumeNiiImage) , dimx , dimy, dimz);
figure;imshow(resizeCt(:,:,3));
newCtImage = double(resizeCt);
nii = make_nii(newCtImage);
save_nii(nii ,newName);
gzip(newName);

end
