%------------------------------------------------------------------------- 
% Wake Forest University Health Sciences
% Date: Sep. 26, 2016
% Routine: MultiSlicesGPU.m
%
% Authors:
%   Rui Liu    (Wake Forest University Health Sciences)
%
% Organization: 
%   Wake Forest University Health Sciences
%
% Aim:
%   This is a high level Matlab/Freemat wrapper function for the distance
%   driven (DD) model and pseudo DD model based projection and
%   backprojection process in a multi-slices geometry with multi GPUs.
%
% Inputs/Outputs:
%
%   Method : a string parameter takeing the following options:
%             'DD_PROJ'       :       Distance driven based projection
%             'DD_BACK'       :       Distance driven based backprojection
%             'PD_PROJ'       :       Pseudo DD based projection
%             'PD_BACK'       :       Pseudo DD based backprojection
%   input : if Method == DD_PROJ or Method == PD_PROJ, the input is the
%   volume in size [SLN, XN, YN] order. if Method == DD_BACK or Method ==
%   PD_BACK, the input is the projection data in size [SLN, DNU, PN] order.
%
%   cfg : Contains geometric parameters of scanning geometry and dimensions
%   of the image volume for reconstruction. The required configuration will
%   be as the following:
%           Source2IsoDis -- Source to iso center distance
%           Source2DetDis -- Source to detector distance
%           imgXCenter -- x position of the image center
%           imgYCenter -- y position of the image center 
%           XN -- reconstructed image size along X direction
%           YN -- reconstructed iamge size along Y direction
%           SLN -- slice number of the volume
%           pixelSize -- size of the pixel 
%           NumberOfDetChannelPerRow -- Number of detector cell along
%           channel direction
%           NumberOfViewPerRotation -- Number of views per rotation
%           DetCellPitch -- detector cell size along channel direction
%           IsoChannelLocation -- Iso center index of the detector

% view_ind
%	optional. default to project all views specified in cfg
%	This allows to speicify a subset of views to project
%
% mask
%	Optional. a 2D binary mask may be provided so that certain region in the image will be ignored from projection
%
% History:
% 	2015-07-24 FUL, unified the wrapper of fw and bk dd3 projectors
%
% TODO:
%	verify compatibility with freemat
%	verify a 2D slice image can be handled (currently 2D case is not carefully handled)
%	Verify/clean up some legacy code for detector upsampling. This feature is not currently supported
%-------------------------------------------------------------------------
function [output] = MultiSlicesGPU(Method, input, cfg, views, mask, startIdx, gpuNum)

sid = single(cfg.Source2IsoDis);
sdd = single(cfg.Source2DetDis);
imgXCenter = single(cfg.imgXCenter);
imgYCenter = single(cfg.imgYCenter);
XN = int32(cfg.XN);
YN = int32(cfg.YN);
SLN = int32(cfg.SLN);
dx = single(cfg.pixelSize);
DNU = int32(cfg.NumberOfDetChannelPerRow);
PN = int32(cfg.NumberOfViewPerRotation);

x0 = single(0);
y0 = single(sid);

% Calculate xds and yds
stepTheta = atan(cfg.DetCellPitch * 0.5 / sdd) * 2.0;
ii = double((0 : (DNU-1)))';
jj = (ii - cfg.IsoChannelLocation) * stepTheta;
xds = single(sin(jj) .* sdd);
yds = single(sid - cos(jj) .* sdd);

[shouldSLN, shouldX, shouldY] = size(input);
if strcmp(Method, 'DD_PROJ')
    method = 0;
    if(shouldSLN ~= SLN || shouldX ~= XN || shouldY ~= YN)
        disp('The dimension did not match');
        return;
    end
else if strcmp(Method, 'DD_BACK')
        method = 1;
        
        if(shouldSLN ~= SLN || shouldX ~= DNU || shouldY ~= PN)
            disp('The dimension did not match');
            return;
        end
    else if strcmp(Method, 'PD_PROJ')
            method = 2;
            if(shouldSLN ~= SLN || shouldX ~= XN || shouldY ~= YN)
                disp('The dimension did not match');
                return;
            end
        else if strcmp(Method, 'PD_BACK')
                method = 3;
                if(shouldSLN ~= SLN || shouldX ~= DNU || shouldY ~= PN)
                    disp('The dimension did not match');
                    return;
                end
            else
                disp('UNKNOWN PROJECTION/BACKPROJECTION METHOD');
                return;
            end
        end
    end
end


[shouldSLN, shouldPN] = size(views);
if(shouldSLN ~= SLN || shouldPN ~= PN)
    disp('Views did not match');
    return;
end


output = multiSlicesGPU_mex(int32(method), x0, y0, xds, yds, ...
    DNU, SLN, imgXCenter, imgYCenter, XN, YN, dx, single(views), ...
    PN, uint8(mask), int32(startIdx), int32(gpuNum), single(input));


