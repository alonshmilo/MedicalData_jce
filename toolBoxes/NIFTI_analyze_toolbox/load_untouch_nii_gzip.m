%  
%  Load NIFTI dataset, but not applying any appropriate affine
%  geometric transform or voxel intensity scaling. NIFTI dataset is
%  unzipped before load_untouch_nii is called when filename ends with '.gz'. 
%
%  Although according to NIFTI website, all those header information are
%  supposed to be applied to the loaded NIFTI image, there are some
%  situations that people do want to leave the original NIFTI header and
%  data untouched. They will probably just use MATLAB to do certain image
%  processing regardless of image orientation, and to save data back with
%  the same NIfTI header.
%
%  Since this program is only served for those situations, please use it
%  together with "save_untouch_nii.m", and do not use "save_nii.m" or
%  "view_nii.m" for the data that is loaded by "load_untouch_nii.m". For
%  normal situation, you should use "load_nii.m" instead.
%  
%  Usage: nii = load_untouch_nii(filename, [img_idx], [dim5_idx], [dim6_idx], [dim7_idx])
%  
%  filename  - 	NIFTI or ANALYZE file name.
%  
%  img_idx (optional)  -  a numerical array of image volume indices.
%	Only the specified volumes will be loaded. All available image
%	volumes will be loaded, if it is default or empty.
%
%	The number of images scans can be obtained from get_nii_frame.m,
%	or simply: hdr.dime.dim(5).
%
%  dim5_idx (optional)  -  a numerical array of 5th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  dim6_idx (optional)  -  a numerical array of 6th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  dim7_idx (optional)  -  a numerical array of 7th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  Returned values:
%  
%  nii structure:
%
%	hdr -		struct with NIFTI header fields.
%
%	filetype -	Analyze format .hdr/.img (0); 
%			NIFTI .hdr/.img (1);
%			NIFTI .nii (2)
%           NIFTI .nii.gz (2)
%
%	fileprefix - 	NIFTI filename without extension.
%
%	machine - 	machine string variable.
%
%	img - 		3D (or 4D) matrix of NIFTI data.
%
% GZIP Extension: Wolf-Dieter Vogl

function [ nii ] = load_untouch_nii_gzip( filename, img_idx, dim5_idx, dim6_idx, dim7_idx )
%LOAD_UNTOUCH_NII_GZIP Summary of this function goes here
%   Detailed explanation goes here

   if ~exist('img_idx','var') | isempty(img_idx)
      img_idx = [];
   end

   if ~exist('dim5_idx','var') | isempty(dim5_idx)
      dim5_idx = [];
   end

   if ~exist('dim6_idx','var') | isempty(dim6_idx)
      dim6_idx = [];
   end

   if ~exist('dim7_idx','var') | isempty(dim7_idx)
      dim7_idx = [];
   end

delGZ=false;
[pathstr, name, ext] = fileparts (filename);
temp_name = tempname;
if ((strcmpi(ext,'.gz') ~= 0 ) )
    filenames=gunzip( filename, temp_name );

    filename=filenames{1};
    [targetfolder, ~,~] = fileparts(filename);

    delGZ=true;
    
elseif(~exist( filename, 'file' ) && ...
        exist( [filename '.gz'], 'file' ))
    
    filenames=gunzip( [filename '.gz'], temp_name );
    filename=filenames{1};
    [targetfolder, ~,~] = fileparts(filename);

    delGZ=true;
end

try 
    nii = load_untouch_nii(filename, img_idx, dim5_idx, dim6_idx, dim7_idx);
catch exception
    if delGZ
        rmdir( targetfolder ,'s');    
    end
    rethrow (exception);
end

if delGZ
    rmdir( targetfolder, 's' );
end
