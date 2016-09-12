--[[
  MKimiSH:
  Receive a model, then use the model to predict some *whole* images
  and output gifs.
]]

require 'image'
require 'nn'
require 'paths'
disp = require 'display'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  predLen = 40,            -- number of samples to produce
  batchSize = 1,           -- at test time always = 1
  inputLen = 1,           -- batchSize = predLen / inputLen
  net = '',              -- path to the generator network
  initName = '',
  name = 'test1',        -- name of the experiment and prefix of file saved
  gpu = 3,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
  nc = 3,                -- # of channels in input
  display = 1,           -- Display image: 0 = false, 1 = true
  loadSize = 360,          -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
  fineSize = 128,        -- size of random crops
  nThreads = 4,          --M FIXED!! -- # of data loading threads to use
  maskName = 'maskppp.png',
  maskValue = 110/255,   --M the init value to fill in mask region
  manualSeed = 0,        -- 0 means random seed
  overlapPred = 0,       -- overlapping edges of center with context

  -- Extra Options:
  noiseGen = 0,          -- 0 means false else true; only works if network was trained with noise too.
  noisetype = 'normal',  -- type of noise distribution (uniform / normal)
  nz = 100,              -- length of noise vector if used
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.noiseGen == 0 then opt.noiseGen = false end
assert(opt.predLen%opt.inputLen == 0, 'I don\'t do padding in time dim')
opt.batchSize = opt.predLen / opt.inputLen
opt.name = opt.name .. '_prl' .. opt.predLen .. '_inl' .. opt.inputLen .. '_ldsz' .. opt.loadSize
opt.data = os.getenv('DATA_ROOT') or 'dataset/target'
assert(opt.data)
opt.withInit = opt.initName ~= ''

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 50000)
end
-- print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

opt.name = opt.name .. '_' .. opt.manualSeed


-- load Context-Encoder
assert(opt.net ~= '', 'provide a generator model')
opt.net = 'checkpoints/' .. opt.net
net = util.load(opt.net, opt.gpu)
net:evaluate()

-- load initializer
local netI
if opt.withInit then
  netI=util.load('checkpoints/' .. opt.initName)
  netI:evaluate()
end

local ncimage = opt.nc
local ncinput = opt.nc * opt.inputLen
local nc = opt.nc * opt.predLen --M
local nz = opt.nz

-- initialize variables
input_image = torch.Tensor(opt.batchSize, ncimage*opt.inputLen, opt.fineSize, opt.fineSize)
local noise
if opt.noiseGen then
    noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
    if opt.noisetype == 'uniform' then
        noise:uniform(-1, 1)
    elseif opt.noisetype == 'normal' then
        noise:normal(0, 1)
    end
end

-- port to GPU
if opt.gpu > 0 then
    require 'cunn'
    if pcall(require, 'cudnn') then
        print('Using CUDNN !')
        require 'cudnn'
        net = util.cudnn(net)
        if opt.withInit then netI=util.cudnn(netI) end
    end
    net:cuda()
    if opt.withInit then netI:cuda() end
    input_image = input_image:cuda()
    if opt.noiseGen then
        noise = noise:cuda()
    end
else
   net:float()
end
-- print(net)

--M no threading for testing here.
local orih, oriw = 360, 480
local inh, inw = opt.loadSize, opt.loadSize * oriw / orih
local outh, outw = math.ceil(inh/opt.fineSize) * opt.fineSize, math.ceil(inw/opt.fineSize) * opt.fineSize
local mask = image.load(opt.maskName)
print(mask:max(), mask:min())
mask = torch.expand(mask, 3, mask:size(2), mask:size(3)):byte()
function loadImages()
  local prx = ''
  local ext = {'_hd1.jpg', '_hd2.jpg', '_hd3.jpg', '_hd4.jpg'}
  local nums = {41573, 41144, 41432, 34376}
  local base = opt.data .. '/images/'
  local scMask = image.scale(mask, inw, inh);
  scMask = scMask:gt(0.3) -- scale mask to avoid boundary effect
  if not paths.dirp(base) then -- train set
    base = opt.data .. '/gt/'
    -- prx = 'train_'
    -- ext = {'_hd1_hd1.jpg', '_hd2_hd2.jpg', '_hd3_hd3.jpg', '_hd4_hd4.jpg'}
  end
  local oriImages = torch.Tensor(opt.predLen, ncimage, inh, inw)
  local images = torch.zeros(opt.predLen, ncimage, outh, outw)
  local vid = torch.random(4)
  local idx = torch.random(nums[vid] - opt.predLen)
  for i = 1, opt.predLen do
    local imName = base .. prx .. string.format('%07d', idx+i) .. ext[vid]
    local im = image.load(imName)
    -- im:maskedFill(mask, 110/255)
    im = image.scale(im, inw, inh)
    im:maskedFill(scMask, opt.maskValue)
    oriImages[i] = image.scale(im:clone(), inw, inh)
  end
  images[{{}, {}, {1, inh}, {1, inw}}] = oriImages:clone() -- padding bottom right
  -- images[{{}, {}, {-inh, -1}, {-inw, -1} }] = oriImages:clone() -- padding top left
  images:mul(2):add(-1)
  -- disp.image(oriImages[opt.predLen-1], {win=77, name='imagess'})
  return images:view(opt.predLen*ncimage, outh, outw):clone()
end

local fullImages  = loadImages()
-- print(fullImages:size())
local outImages = torch.Tensor(opt.predLen, ncimage, outh, outw)
local inpaintImages = fullImages:clone()

-- pass through network every block and save image in outImages
-- print(outh, outw)

local mid_mask = image.scale(mask, inw, inh)
mid_mask = mid_mask:gt(0.3)
local tmp = torch.ByteTensor(mid_mask:size(1), outh, outw)
tmp[{{}, {1, inw}, {1, inh}}] = mid_mask
mid_mask = tmp:clone()
for h=1, outh, opt.fineSize do
  for w=1, outw, opt.fineSize do
    -- print(h, w)
    local input_image_patch = torch.Tensor(input_image:size())
    for fr=1, nc, ncinput do
      --print('fr=', fr)
      local idx = (fr+ncinput-1) / ncinput
      local tmp_patch = fullImages[{{fr, fr+ncinput-1}, {h, h+opt.fineSize-1}, {w, w+opt.fineSize-1}}]
      if h==1 and (w==1 or w==opt.fineSize+1 or w==opt.fineSize*2+1) then
        tmp_patch = image.vflip(tmp_patch)
        -- print('flipped!')
      end
      input_image_patch[idx]:copy(tmp_patch)
    end
    --disp.image(input_image_patch, {win=20})
    --local input_image_patch = fullImages[{{}, {h, h+opt.fineSize-1}, {w, w+opt.fineSize-1}}]
    --  :clone():view(opt.batchSize, opt.inputLen*ncimage, opt.fineSize, opt.fineSize)
    input_image:copy(input_image_patch)
    -- print(input_image:size(), input_image:type())
    local out_image
    if not opt.withInit then
      out_image = net:forward(input_image)
    else
      local mid_image = netI:forward(input_image)
      local tmp_mask = mid_mask[{{}, {h, h+opt.fineSize-1}, {w, w+opt.fineSize-1}}]
      tmp_mask = tmp_mask:cuda()

      -- mid_mask = mid_mask:cuda()
      inpainter = paths.dofile('inpaint_utils.lua')
      mid_image = inpainter.fillIn(input_image, tmp_mask, mid_image)
      out_image = net:forward(mid_image)
    end
    -- out_image = image.vflip(out_image:float())

    -- disp.image(out_image, {win=22})
    if h==1 and (w==1 or w==opt.fineSize+1 or w==opt.fineSize*2+1) then
      for fr=1, nc, ncinput do
        --print('fr=', fr)
        local idx = (fr+ncinput-1) / ncinput
        out_image[idx]:copy(image.vflip(out_image[idx]:float()))
      end
    end

    outImages[{{}, {}, {h, h+opt.fineSize-1}, {w, w+opt.fineSize-1}}]:copy(
      out_image:view(opt.predLen, ncimage, opt.fineSize, opt.fineSize))
  end
end

inpaintImages = inpaintImages:viewAs(outImages)
mask = image.scale(mask, inw, inh)
local padmask = torch.zeros(outImages[1]:size())
padmask = padmask:byte()
padmask[{{}, {1, mask:size(2)}, {1, mask:size(3)}}] = mask:clone() -- pad bottom right
-- padmask[{{}, {-mask:size(2), -1}, {-mask:size(3), -1}}] = mask:clone()

for i=1, opt.predLen do
  --print(outImages[i]:size(), padmask:size(), padmask:type())
  local maskedout = outImages[i]:maskedSelect(padmask)
  --local maskedout = torch.FloatTensor()
  maskedout:maskedSelect(outImages[i], padmask)
  inpaintImages[i]:maskedCopy(padmask, maskedout)
end

outImages:add(1):mul(0.5)
fullImages:add(1):mul(0.5)
inpaintImages:add(1):mul(0.5)

if not paths.dirp(opt.name) then paths.mkdir(opt.name) end
-- save every single image
for fr=1, opt.predLen do
  local imName = opt.name .. string.format('/pred_%d.png', fr)
  -- disp.image(outImages[fr], {win=11, name='test'})
  image.save(imName, outImages[fr])
end
for fr=1, opt.predLen do
  local imName = opt.name .. string.format('/inpaint_%d.png', fr)
  -- disp.image(inpaintImages[fr], {win=12, name='inpaint'})
  image.save(imName, inpaintImages[fr])
end
for fr=1, opt.predLen do
  local imName = opt.name .. string.format('/orig_%d.png', fr)
  -- disp.image(fullImages[{{fr*3-2, fr*3}}], {win=13, name='orig'})
  image.save(imName, fullImages[{{fr*3-2, fr*3}}])
end

local delay_gif = 10

-- save as one gif image
os.execute('convert $(for ((a=1; a<'..opt.predLen..
    '; a++)); do printf -- "-delay '..delay_gif..' '.. opt.name ..
    '/pred_%s.png " $a; done;) ' .. opt.name  .. '_result.gif')
    
os.execute('convert $(for ((a=1; a<'..opt.predLen..
    '; a++)); do printf -- "-delay '..delay_gif..' '.. opt.name ..
    '/inpaint_%s.png " $a; done;) ' .. opt.name  .. '_inpaint.gif')
    
os.execute('convert $(for ((a=1; a<'..opt.predLen..
    '; a++)); do printf -- "-delay '..delay_gif..' '.. opt.name ..
    '/orig_%s.png " $a; done;) ' .. opt.name  .. '_orig.gif') 