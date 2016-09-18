--[[
    MKimiSH: This version samples a piece of video rather than random images
    Added random block masking!!
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).

    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]

    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
--paths.dofile('dataset_norand.lua')
paths.dofile('dataset_wholeim.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT')
if not paths.dirp(opt.data) then
    error('Did not find directory: ', opt.data)
end
--M Check for mask
if not paths.filep(opt.maskName) then
  error('Did not find mask: ', opt.maskName)
end
mask = image.load(opt.maskName)
mask = mask:byte()
assert(mask:max() <= 1)

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local nc = opt.nc
local loadSize   = {nc, opt.loadSize}
local sampleSize = {nc, opt.fineSize}

--M used
local function loadImage(path)
   assert(#path==1, 'I know path is a table and it should be length 1')
   local input = image.load(path[1], nc, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   if loadSize[2]>0 then
     local iW = input:size(3)
     local iH = input:size(2)
     if iW < iH then
        input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
     else
        input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
     end
   elseif loadSize[2]<0 then
    local scalef = 0
     if loadSize[2] == -1 then
       scalef = torch.uniform(0.5,1.5)
     else
       scalef = torch.uniform(1,3)
     end
     local iW = scalef*input:size(3)
     local iH = scalef*input:size(2)
     input = image.scale(input, iH, iW)
   end
   mask = image.scale(mask, input:size(3), input:size(2))
   return input
end

--M load images
--M not used
local function loadContImages(paths)
  local inputTable = {}
  for i=1, #paths do
    local inputoneim = image.load(paths[i], nc, 'float')
    inputTable[i] = inputoneim
  end
  local input = torch.Tensor(opt.predLen, nc, inputTable[1]:size(2), inputTable[1]:size(3))
  for i=1, #paths do
    input[i]:copy(inputTable[i])
  end
  input = input:view(opt.predLen*nc, input:size(3), input:size(4))
  -- print(input:size())
   -- local input = image.load(path, nc, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   if loadSize[2]>0 then
     local iW = input:size(3)
     local iH = input:size(2)
     if iW < iH then
        input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
     else
        input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
     end
   elseif loadSize[2]<0 then
    local scalef = 0
     if loadSize[2] == -1 then
       scalef = torch.uniform(0.5,1.5)
     else
       scalef = torch.uniform(1,3)
     end
     local iW = scalef*input:size(3)
     local iH = scalef*input:size(2)
     input = image.scale(input, iH, iW)
   end
   --input = input:view(opt.predLen, nc, input:size(2), input:size(3))
   mask = image.scale(mask, input:size(3), input:size(2))
   return input
end

--M maybe the parameter can be tuned
--M not used??
local function randomBlockMask(im)
  assert(im:dim() == 3)
  local maskout = torch.Tensor(im:size())
  maskout:byte()
  local h, w = im:size(2), im:size(3)
  local blockSize = math.floor(h/6)  --M square block
  local maxBlocks = 10 -- maybe tunable.. cover 30% at most.
  local nBlocks = torch.random(2, maxBlocks)
  for i=1, nBlocks do
    local tlx, tly = torch.random(3, w-blockSize-2), torch.random(3, h-blockSize-2)
    -- don't be too sidely.. x-w, y-h
    maskout[{{}, {tly, tly+blockSize-1}, {tlx, tlx+blockSize-1}}] = 1
    im[{{}, {tly, tly+blockSize-1}, {tlx, tlx+blockSize-1}}] = opt.maskValue
  end
  return im:clone(), maskout
end

-- channel-wise mean and std. Calculate or load  them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- trainHook_obsolete is the one used in training with one patch at a time!
local trainHook= function(self, path, withMask)
  collectgarbage()
  local input = loadImage(path)
  local iW, iH = input:size(3), input:size(2)
  local arrw, arrh = opt.array_w, opt.array_h
  local outw, outh = 2, 2  -- for the output of the network, not *out* here
  local ss = sampleSize[2]

  -- no random crop now!, but do it on 0908, 233!
  assert(withMask==1, 'should output mask')

  -- crop and pad together!
  local steph = math.floor((iH-ss)/(arrh-1))
  local stepw = math.floor((iW-ss)/(arrw-1))

  -- three things should be outputted
  local out = torch.Tensor(opt.nc*outw*outh, ss, ss)
  local maskout = torch.Tensor(opt.nc*outw*outh, ss, ss)
  local masked = torch.Tensor(opt.nc*arrw*arrh, ss, ss)

  -- mask input
  local expandedmask = mask:expand(input:size())
  local maskedinput = input:clone()
  maskedinput:maskedFill(expandedmask, opt.maskValue)

  -- do random crop
  local maxcrop_w, maxcrop_h = 100, 70 -- do random crop but not too much
  local crop_w, crop_h = torch.random(maxcrop_w), torch.random(maxcrop_h)
  local tmpinput, tmpmask, tmpmaskedinput =
    torch.zeros(input:size()), torch.zeros(expandedmask:size()), torch.zeros(maskedinput:size())
  tmpmask = tmpmask:byte()
  tmpinput[{{}, {1, iH-crop_h+1}, {1, iW-crop_w+1}}]:copy(input[{{},{crop_h, iH},{crop_w, iW}}])
  tmpmask[{{}, {1, iH-crop_h+1}, {1, iW-crop_w+1}}]:copy(expandedmask[{{},{crop_h, iH},{crop_w, iW}}])
  tmpmaskedinput[{{}, {1, iH-crop_h+1}, {1, iW-crop_w+1}}]:copy(maskedinput[{{},{crop_h, iH},{crop_w, iW}}])

  -- do hflipping
  if torch.uniform() > 0.6 then
    tmpmask = image.hflip(tmpmask)
    tmpinput = image.hflip(tmpinput)
    tmpmaskedinput = image.hflip(tmpmaskedinput)
  end

  -- copy it back so I don't have to change the following code
  input = tmpinput:clone()
  expandedmask = tmpmask:clone()
  maskedinput = tmpmaskedinput:clone()

  local topleftpatch = input[{{},{1,ss},{1,ss}}]:clone()
  if torch.mean(topleftpatch) < 0.1 then
    if torch.uniform() > 0.1 then
      return nil
    end
  end

  local cntpatch = -2
  for h=1,iH-ss+1,steph do
    for w=1,iW-ss+1,stepw do
      -- print(h, w)
      cntpatch = cntpatch + 3
      local maskedpatch = maskedinput[{{}, {h,h+ss-1}, {w,w+ss-1}}]
      local h1, w1 = math.floor(h/steph), math.floor(w/stepw)
      masked[{{cntpatch, cntpatch+2},{},{}}]:copy(maskedpatch)
      if h1<=1 and w1<=1 then -- output top left 4 patches
        local outpatch = input[{{}, {h,h+ss-1}, {w,w+ss-1}}]:clone()
        local maskpatch = expandedmask[{{}, {h,h+ss-1}, {w,w+ss-1}}]
        local idx = (h1*2+w1)*opt.nc + 1 --(0,1)->4
        out[{{idx, idx+2}, {}, {}}]:copy(outpatch)
        maskout[{{idx, idx+2},{}, {}}]:copy(maskpatch)
      end
    end
  end
  out:mul(2):add(-1)
  masked:mul(2):add(-1)
  return out, maskout, masked
end

-- function to load the image, jitter it appropriately (random crops etc.)
--M read many images rather than one.
local trainHook_obsolete= function(self, path, withMask)
   collectgarbage()
   local input = loadContImages(path) -- in this case it should be *paths*
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2];
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   if torch.mean(out) < 0.1 then
     --print(torch.mean(out))
     if torch.uniform() > 0.05 then
       return nil
     end
   end

   local maskout
   local masked
   if withMask then
     -- print(mask:size(), iH, iW)
     assert(mask:size(2) == iH and mask:size(3) == iW, 'maskSize inconsistent!')
     maskout = image.crop(mask, w1, h1, w1 + oW, h1 + oH)
     maskout = torch.expand(maskout, out:size(1), maskout:size(2), maskout:size(3))
     masked = out:clone()
     if maskout:max() > 0.5 then --M mask is not all black
       masked:maskedFill(maskout, opt.maskValue)
     else
       masked, maskout = randomBlockMask(masked)
     end
   end
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then
     out = image.hflip(out);
     if withMask then
       masked = image.hflip(masked)
       maskout= image.hflip(maskout)
     end
   end
   out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   if withMask then
     masked:mul(2):add(-1)
     return out, maskout, masked
   end
   return out:view(opt.predLen, nc, out:size(2), out:size(3))
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {nc, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {nc, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {opt.data},
      loadSize = {nc, loadSize[2], loadSize[2]},
      sampleSize = {nc, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
