function resizeCt = resize(filepath,newName)

CtImageInRegularSize = load_untouch_nii_gzip (filepath);
volumeNiiImage = double(CtImageInRegularSize.img);
figure;imshow(volumeNiiImage(:,:,3));
resizeCt = CTresize2((volumeNiiImage) , 240 , 240, 155);
figure;imshow(resizeCt(:,:,3));
nii = make_nii(resizeCt);
save_nii(nii ,newName);
gzip(newName);

end

