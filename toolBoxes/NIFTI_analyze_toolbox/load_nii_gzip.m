%  Load NIFTI or ANALYZE dataset. Support *.nii, *.nii.gz and *.hdr/*.img
%  file extension. If file extension is not provided, *.hdr/*.img will
%  be used as default.
%
%  A subset of NIFTI transform is included. For non-orthogonal rotation,
%  shearing etc., please use 'reslice_nii.m' to reslice the NIFTI file.
%  It will not cause negative effect, as long as you remember not to do
%  slice time correction after reslicing the NIFTI file. Output variable
%  nii will be in RAS orientation, i.e. X axis from Left to Right,
%  Y axis from Posterior to Anterior, and Z axis from Inferior to
%  Superior.
%  
%  Usage: nii = load_nii(filename, [img_idx], [dim5_idx], [dim6_idx], ...
%			[dim7_idx], [old_RGB], [tolerance], [preferredForm])
%  
%  filename  - 	NIFTI or ANALYZE file name.
%  
%  img_idx (optional)  -  a numerical array of 4th dimension indices,
%	which is the indices of image scan volume. The number of images
%	scan volumes can be obtained from get_nii_frame.m, or simply
%	hdr.dime.dim(5). Only the specified volumes will be loaded. 
%	All available image volumes will be loaded, if it is default or
%	empty.
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
%  old_RGB (optional)  -  a scale number to tell difference of new RGB24
%	from old RGB24. New RGB24 uses RGB triple sequentially for each
%	voxel, like [R1 G1 B1 R2 G2 B2 ...]. Analyze 6.0 from AnalyzeDirect
%	uses old RGB24, in a way like [R1 R2 ... G1 G2 ... B1 B2 ...] for
%	each slices. If the image that you view is garbled, try to set 
%	old_RGB variable to 1 and try again, because it could be in
%	old RGB24. It will be set to 0, if it is default or empty.
%
%  tolerance (optional) - distortion allowed in the loaded image for any
%	non-orthogonal rotation or shearing of NIfTI affine matrix. If 
%	you set 'tolerance' to 0, it means that you do not allow any 
%	distortion. If you set 'tolerance' to 1, it means that you do 
%	not care any distortion. The image will fail to be loaded if it
%	can not be tolerated. The tolerance will be set to 0.1 (10%), if
%	it is default or empty.
%
%  preferredForm (optional)  -  selects which transformation from voxels
%	to RAS coordinates; values are s,q,S,Q.  Lower case s,q indicate
%	"prefer sform or qform, but use others if preferred not present". 
%	Upper case indicate the program is forced to use the specificied
%	tranform or fail loading.  'preferredForm' will be 's', if it is
%	default or empty.	- Jeff Gunter
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
%
%	fileprefix - 	NIFTI filename without extension.
%
%	machine - 	machine string variable.
%
%	img - 		3D (or 4D) matrix of NIFTI data.
%
%	original -	the original header before any affine transform.
%  
%  Part of this file is copied and modified from:
%  http://www.mathworks.com/matlabcentral/fileexchange/1878-mri-analyze-tools
%  
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%  
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
% 
% GZIP extension: Wolf-Dieter Vogl

function [ nii ] = load_nii_gzip(filename, img_idx, dim5_idx, dim6_idx, dim7_idx, ...
			old_RGB, tolerance, preferredForm)
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
   if ~exist('old_RGB','var') | isempty(old_RGB)
      old_RGB = [];
   end
      if ~exist('tolerance','var') | isempty(tolerance)
      tolerance = [];
      end
   if ~exist('preferredForm','var') | isempty(preferredForm)
      preferredForm = [];
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
    nii = load_nii(filename, img_idx, dim5_idx, dim6_idx, dim7_idx, old_RGB, tolerance, preferredForm);
catch exception
    if delGZ
        rmdir( targetfolder ,'s');    
    end
    rethrow (exception);
end

if delGZ
    rmdir( targetfolder, 's' );
end
