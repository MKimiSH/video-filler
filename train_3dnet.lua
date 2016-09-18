--[[ modified by MKimiSH, modified part marked with --M
  network uses 3D-convolution(strided in time dimension):
  4x23x128x128 input -> 3x23x128x128 output
  4x23x128x128, 32x23x64x64, 64x23x32x32, 192x11x16x16, 512x5x8x8
  and reverse to form netG
  netD: 384x5x8x8 -> fc 10000-> 1
]]

require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
util = paths.dofile('util.lua')
inpainter = paths.dofile('inpaint_utils.lua')

opt = {
   batchSize = 32,         -- number of video samples to produce
   loadSize = 360,         -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
   --M: loadSize should not be -1 because there are two images to read simultaneously!!
   fineSize = 128,         -- size of random crops
   nBottleneck = 8192,      -- #  of dim for bottleneck of encoder
   nef = 32,               -- #  of encoder filters in first conv layer
   ngf = 32,               -- #  of gen filters in first conv layer
   ndf = 32,               -- #  of discrim filters in first conv layer
   ncin = 4,                 -- # of channels in input
   ncout = 3,               -- # of channels in output
   predLen = 23,            --M # of frames in a clip
   wtl2 = 0.999,           -- 0 means don't use else use with this weight --M this weight is necessary
   weight_nomask = 0.05,   --M non-mask region weight
   wtgdl = 0,              --M gdl criterion weight
   overlapPred = 0,        -- overlapping edges --M: for arbitrary shape this should be 0
   nThreads = 1,           -- #  of data loading threads to use --M I don't know if multithreading can correctly return successive frames
   niter = 500,            -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = 25600,         -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_iter = 50,      -- # number of iterations after which display is updated
   display_port = 8000, 
   gpu = 0,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'train1',        -- name of the experiment you are running
   manualSeed = 0,         -- 0 means random seed
   maskName = 'maskppp.png',  -- name of mask, not only square --M
   maskValue = 110/255,
   loadName = '',
   loadIter = 0,

   withInit = 0,
   initName = '',
   -- Extra Options:
   conditionAdv = 0,       -- 0 means false else true
   noiseGen = 0,           -- 0 means false else true
   noisetype = 'normal',   -- uniform / normal
   nz = 100,               -- #  of dim for Z
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.conditionAdv == 0 then opt.conditionAdv = false end
opt.conditionAdv = false --M in my case this is never true!!!!!!!
if opt.noiseGen == 0 then opt.noiseGen = false end

--[[M In this setting, opt.conditionAdv should be 0, opt.noiseGen should be 0,
]]--

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('datavid/data_3d.lua')
local data = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

---------------------------------------------------------------------------
-- Initialize network variables
---------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   elseif name:find('Linear') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   end
end
-- because only nil and false are FALSE, so "if *var*" can be used gracefully

local ncout = opt.ncout * opt.predLen
local ncin = opt.ncin * opt.predLen --M
local nz = opt.nz
local nBottleneck = opt.nBottleneck
local ndf = opt.ndf
local ngf = opt.ngf
local nef = opt.nef
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local volconv = nn.VolumetricConvolution
local volbnorm = nn.VolumetricBatchNormalization
local volfconv = nn.VolumetricFullConvolution

--volconv(ncin, ncout, kt, kw, kh, dt, dw, dh, padt, padw, padh)
--volbnorm(N)
--volfconv(ncin, ncout, kt, kw, kh, dt, dw, dh, padt, padw, padh)

---------------------------------------------------------------------------
-- Generator net
---------------------------------------------------------------------------
-- Encode Input Context to noise (architecture similar to Discriminator)
local netE = nn.Sequential()
-- input: nc x 23x128x128
netE:add(volconv(ncin, nef, 1,4,4,  1,2,2,  0,1,1))
netE:add(nn.LeakyReLU(0.2, true))
-- now: nef x 23x64x64
netE:add(volconv(nef, nef*2, 1,4,4,  1,2,2,  0,1,1))
netE:add(volbnorm(nef*2)):add(nn.LeakyReLU(0.2, true))
-- now: nef*2 x 23x32x32
netE:add(volconv(nef*2, nef*4, 3,3,3,  1,1,1,  1,1,1))
netE:add(volbnorm(nef*4)):add(nn.LeakyReLU(0.2, true))
-- now: nef*4 x 23x32x32
netE:add(volconv(nef*4, nef*6,  3,4,4,  2,2,2,  0,1,1))
netE:add(volbnorm(nef*6)):add(nn.LeakyReLU(0.2, true))
-- now: nef*6 x 11x16x16
netE:add(volconv(nef*6, nef*12, 3,4,4,  2,2,2,  0,1,1))
netE:add(volbnorm(nef*12)):add(nn.LeakyReLU(0.2, true))
-- now: nef*12 x 5x8x8
netE:add(volconv(nef*12, nef*20, 3,4,4,  1,1,1,  0,0,0))
-- now: nef*20 x 3x5x5

local netG = nn.Sequential()
local nz_size = nBottleneck
if opt.noiseGen then --M should not be true.
    local netG_noise = nn.Sequential()
    -- input is Z: (nz) x 1 x 1, going into a convolution
    netG_noise:add(SpatialConvolution(nz, nz, 1, 1, 1, 1, 0, 0))
    -- state size: (nz) x 1 x 1

    local netG_pl = nn.ParallelTable();
    netG_pl:add(netE)
    netG_pl:add(netG_noise)

    netG:add(netG_pl)
    netG:add(nn.JoinTable(2))
    netG:add(SpatialBatchNormalization(nBottleneck+nz)):add(nn.LeakyReLU(0.2, true))
    -- state size: (nBottleneck+nz) x 1 x 1

    nz_size = nBottleneck+nz
else
    netG:add(netE)
    netG:add(SpatialBatchNormalization(nef*20)):add(nn.LeakyReLU(0.2, true))

    nz_size = nBottleneck
end

-- Decode noise to generate image
-- now: nef*20 x 3x5x5
netG:add(volfconv(nef*20, ngf*12, 3,4,4,  1,1,1,  0,0,0))
netG:add(volbnorm(ngf*12):add(nn.ReLU(true)))
-- now: ngf*16 x 5x8x8
netG:add(volfconv(ngf*12, ngf*6,  3,4,4,  2,2,2,  0,1,1))
netG:add(volbnorm(ngf*6)):add(nn.ReLU(true))
-- now: ngf*6 x 11x16x16
netG:add(volfconv(ngf*6, ngf*4,  3,3,3,  1,1,1,  1,1,1))
netG:add(volbnorm(ngf*4)):add(nn.ReLU(true))
-- now: ngf*4 x 23x32x32
netG:add(volfconv(ngf*4, ngf*2,  1,4,4,  1,2,2,  0,1,1))
netG:add(volbnorm(ngf*2)):add(nn.ReLU(true))
-- now: ngf*2 x 23x32x32
netG:add(volfconv(ngf*2, ngf,  1,4,4,  1,2,2,  0,1,1))
netG:add(volbnorm(ngf)):add(nn.ReLU(true))
-- now: ngf x 23x64x64
netG:add(volfconv(ngf, ncout,  1,4,4,  1,2,2,  0,1,1))
netG:add(nn.Tanh())
-- now: ncout x 23x128x128

netG:apply(weights_init)
-- first design a network without real implementation then NN:apply somefunctions?


---------------------------------------------------------------------------
-- Adversarial discriminator net
---------------------------------------------------------------------------
local netD = nn.Sequential()

-- input: ncout x 23x128x128
netD:add(volconv(ncout, ndf, 1,4,4,  1,2,2,  0,1,1))
netD:add(nn.LeakyReLU(0.2, true))
-- now: nef x 23x64x64
netD:add(volconv(ndf, ndf*2, 1,4,4,  1,2,2,  0,1,1))
netD:add(volbnorm(ndf*2)):add(nn.LeakyReLU(0.2, true))
-- now: nef*2 x 23x32x32
netD:add(volconv(ndf*2, ndf*4, 3,3,3,  1,1,1,  1,1,1))
netD:add(volbnorm(ndf*4)):add(nn.LeakyReLU(0.2, true))
-- now: nef*4 x 23x32x32
netD:add(volconv(ndf*4, ndf*6,  3,4,4,  2,2,2,  0,1,1))
netD:add(volbnorm(ndf*6)):add(nn.LeakyReLU(0.2, true))
-- now: nef*6 x 11x16x16
netD:add(volconv(ndf*6, ndf*12, 3,4,4,  2,2,2,  0,1,1))
netD:add(volbnorm(nef*12)):add(nn.LeakyReLU(0.2, true))
-- now: nef*16 x 5x8x8
netD:add(volconv(ndf*12, ndf*20, 3,4,4,  1,1,1,  0,0,0))
netD:add(volbnorm(ndf*20)):add(nn.LeakyReLU(0.2, true)))
-- now: nef*20 x 3x5x5
netD:add(volconv(ndf*20, nBottleneck, 3,5,5))
netD:add(nn.Sigmoid())
-- now: nBottleneck x 1x1x1
netD:add(nn.View(1):setNumInputDims(4))
netD:add(nn.Linear(nBottleneck, 1))
netD:add(nn.Sigmoid())

netD:apply(weights_init)

-- load netG, netD
if opt.loadName ~= '' then
  local loadNameD, loadNameG
  assert(opt.loadIter > 0, 'Not want to finetune?')

  if opt.loadName ~= '' then
    loadNameD = './checkpoints/' .. opt.loadName .. '_' .. opt.loadIter .. '_net_D.t7'
    loadNameG = './checkpoints/' .. opt.loadName .. '_' .. opt.loadIter .. '_net_G.t7'
  else
    loadNameD = './checkpoints/' .. opt.name .. '_' .. opt.loadIter .. '_net_D.t7'
    loadNameG = './checkpoints/' .. opt.name .. '_' .. opt.loadIter .. '_net_G.t7'
  end
  assert(paths.filep(loadNameD), 'netD not found')
  assert(paths.filep(loadNameG), 'netG not found')
  netG = util.load(loadNameG, opt.gpu)
  netD = util.load(loadNameD, opt.gpu)
end

-- load Initializer if necessary
local netI
if opt.withInit ~= 0 then
  assert(opt.initName~='', 'Give me an initializer!')
  netI = util.load(opt.initName, opt.gpu)
end

---------------------------------------------------------------------------
-- Loss Metrics
---------------------------------------------------------------------------
local criterion = nn.BCECriterion() -- binary cross-entropy
local criterionMSE, criterionGDL
if opt.wtl2~=0 then
  criterionMSE = nn.MSECriterion()
end
if opt.wtgdl~=0 then
  require 'gdl_criterion' 
  criterionGDL = nn.GDLCriterion(1)
end

---------------------------------------------------------------------------
-- Setup Solver
---------------------------------------------------------------------------
-- Note: In lua, a and b and c == c if a==true and b==true
-- d = a and b or c -- d = b if (a and b)==true; else d = c
print('LR of Gen is ',(opt.wtl2>0 and opt.wtl2<1) and 10 or 1,'times Adv')
optimStateG = {
   learningRate = (opt.wtl2>0 and opt.wtl2<1) and opt.lr*10 or opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

---------------------------------------------------------------------------
-- Initialize data variables
---------------------------------------------------------------------------
local input_ctx_vis = torch.Tensor(opt.batchSize, ncin, opt.predLen, opt.fineSize, opt.fineSize)
--M input_ctx_vis:view(opt.batchSize, opt.predLen, opt.nc, opt.fineSize, opt.fineSize) SHOULD
--  give correctly arranged 2D array of RGB images
local input_ctx = torch.Tensor(opt.batchSize, ncin, opt.predLen, opt.fineSize, opt.fineSize) -- this is the real input, only this should be 4-channel
local input_inpainted = torch.Tensor(opt.batchSize, ncout, opt.predLen, opt.fineSize, opt.fineSize)
local input_mask = torch.Tensor(opt.batchSize, ncout, opt.predLen, opt.fineSize, opt.fineSize)
--M if mask has only one channel, the :maskedSelect won't work.

--[[M in this case, it is not 'center', but the masked area which may be empty.
thus the input can be only the 'input_ctx', and the guidance is input_real,
which is the whole image. This is more like training an denoising autoencoder..
Let's see what will happen.
  One thing: should the nn structure be changed????? =changed
]]--
--[[M local input_real_center
if opt.wtl2~=0 then
    input_real_center = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
  end ]]--
local input_real
if opt.wtl2~=0 then
  input_real = torch.Tensor(opt.batchSize, ncout, opt.predLen, opt.fineSize, opt.fineSize)
end

local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG, errG_l2, errG_gdl
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

if pcall(require, 'cudnn') and pcall(require, 'cunn') and opt.gpu>0 then
    print('Using CUDNN !')
end
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input_ctx_vis = input_ctx_vis:cuda(); input_ctx = input_ctx:cuda(); -- input_center = input_center:cuda()
   input_inpainted = input_inpainted:cuda(); input_mask = input_mask:cuda()
   noise = noise:cuda();  label = label:cuda()
   netG = util.cudnn(netG);     netD = util.cudnn(netD)
   netD:cuda();           netG:cuda();           criterion:cuda();
   if opt.withInit ~=0 then
     netI = util.cudnn(netI)    netI:cuda();
   end
   if opt.wtl2~=0 then
      criterionMSE:cuda(); input_real = input_real:cuda() -- input_real_center = input_real_center:cuda();
   end
   if opt.wtgdl~=0 then
      criterionGDL:cuda()
   end
end
if opt.withInit ~=0 then
  print('NetI:', netI)
end
print('NetG:', netG)
print('NetD:', netD)

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then 
   disp = require 'display'; 
   disp.configure({hostname='127.0.0.1', port=opt.display_port})
end

noise_vis = noise:clone()
if opt.noisetype == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise_vis:normal(0, 1)
end

---------------------------------------------------------------------------
-- Define generator and adversary closures
---------------------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()

   -- train with real
   --M: modify the input when reading data or modify afterwards?
   data_tm:reset(); data_tm:resume()
   --M print('BEFORE GETBATCH')
   local real_ctx, real_full, real_mask = data:getBatch() --M real_ctx is masked out, while real_full is not.

   data_tm:stop()
   input_ctx:copy(real_ctx)
   if opt.wtl2~=0 then
     input_real:copy(real_full)
   end
   label:fill(real_label)
   input_mask:copy(real_mask)
   --[[M: In arbitrary region inpainting, D is never conditioned on context!!!

   local output
   if opt.conditionAdv then
      output = netD:forward({input_ctx,input_center})
   else
      output = netD:forward(input_center)
     end]]--

   if opt.withInit~=0 then
     local fake_init = netI:forward(input_ctx)
     -- print(input_ctx:size(), input_mask:size())
     input_ctx = inpainter.fillIn(input_ctx, input_mask, fake_init)
   end
   local output = netD:forward(input_real)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   if opt.conditionAdv then
      netD:backward({input_ctx,input_real}, df_do)
   else
      netD:backward(input_real, df_do)
   end

   -- train with fake
   if opt.noisetype == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noisetype == 'normal' then
       noise:normal(0, 1)
   end
   local fake
   if opt.noiseGen then
      fake = netG:forward({input_ctx,noise})
   else
      fake = netG:forward(input_ctx)
   end
   --M at not masked areas, input_inpainted should be gt
   
   if opt.weight_nomask == 0 then -- totally mask out the gradient of non-mask area.
     -- input_inpainted:copy(real_full)
     -- local maskedout = fake:maskedSelect(input_mask)
     -- input_inpainted:maskedCopy(input_mask, maskedout)
     input_inpainted = inpainter.fillIn(real_full, input_mask, fake)
   else 
     input_inpainted:copy(fake)
   end
   -- input_inpainted:copy(fake)
   label:fill(fake_label)
   

   local output
   if opt.conditionAdv then
     --M reserve 'input_center' to raise error when conditionAdv ~= false
      output = netD:forward({input_ctx,input_center})
   else
      output = netD:forward(input_inpainted)
   end
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   if opt.conditionAdv then
      netD:backward({input_ctx,input_center}, df_do)
   else
      netD:backward(input_inpainted, df_do)
   end

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward({input_ctx,noise})
   input_center:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward({input_ctx,input_center}) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg
   if opt.conditionAdv then
      df_dg = netD:updateGradInput({input_ctx,input_center}, df_do)
      df_dg = df_dg[2]     -- df_dg[2] because conditional GAN
   else
      df_dg = netD:updateGradInput(input_real, df_do)
   end

   local errG_total = errG
   if opt.wtl2~=0 then
      local df_dg_l2
      if opt.weight_nomask == 0 then
      --M      errG_l2 = criterionMSE:forward(input_center, input_real_center)
        errG_l2 = criterionMSE:forward(input_inpainted, input_real)
      --M      local df_dg_l2 = criterionMSE:backward(input_center, input_real_center)
        df_dg_l2 = criterionMSE:backward(input_inpainted, input_real)
      else
        local lambda = opt.weight_nomask
        local weights = input_mask:mul(1-lambda):add(lambda)
        errG_l2 = criterionMSE:forward(input_inpainted, input_real)
        df_dg_l2 = criterionMSE:backward(input_inpainted, input_real)
        df_dg_l2:cmul(weights)
      end
      
      if opt.overlapPred==0 then --M overlapPred should should should be 0
        if (opt.wtl2>0 and opt.wtl2<1) then
          df_dg:mul(1-opt.wtl2):add(opt.wtl2,df_dg_l2)
          errG_total = (1-opt.wtl2)*errG + opt.wtl2*errG_l2
        else
          df_dg:add(opt.wtl2,df_dg_l2)
          errG_total = errG + opt.wtl2*errG_l2
        end
      --M the *else* SHOULD NOT be executed
      else -- opt.overlapPred > 0
        local overlapL2Weight = 10
        local wtl2Matrix = df_dg_l2:clone():fill(overlapL2Weight*opt.wtl2) -- fill the boundaries with 10*opt.wtl2
        wtl2Matrix[{{},{},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred}}]:fill(opt.wtl2) -- fill the rest part with opt.wtl2
        if (opt.wtl2>0 and opt.wtl2<1) then
          df_dg:mul(1-opt.wtl2):addcmul(1,wtl2Matrix,df_dg_l2)
          errG_total = (1-opt.wtl2)*errG + opt.wtl2*errG_l2
        else
          df_dg:addcmul(1,wtl2Matrix,df_dg_l2)
          errG_total = errG + opt.wtl2*errG_l2
        end
      end
   end
   
   if opt.wtgdl~=0 then
     errG_gdl = criterionGDL:forward(input_inpainted, input_real)
     local df_dg_gdl = criterionMSE:backward(input_inpainted, input_real)
     errG_total = errG_total + opt.wtgdl*errG_gdl
     df_dg:add(opt.wtgdl, df_dg_gdl)
   end

   if opt.noiseGen then
      netG:backward({input_ctx,noise}, df_dg)
   else
      netG:backward(input_ctx, df_dg)
   end

   return errG_total, gradParametersG
end

---------------------------------------------------------------------------
-- Train Context Encoder
---------------------------------------------------------------------------
for epoch = opt.loadIter+1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % opt.display_iter == 0 and opt.display then
          local real_ctx, real_full, real_mask = data:getBatch()
          --[[local real_center = real_ctx[{{},{},{1 + opt.fineSize/4, opt.fineSize/2 + opt.fineSize/4},{1 + opt.fineSize/4, opt.fineSize/2 + opt.fineSize/4}}]:clone() -- copy by value
          real_ctx[{{},{1},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*117.0/255.0 - 1.0
          real_ctx[{{},{2},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*104.0/255.0 - 1.0
          real_ctx[{{},{3},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*123.0/255.0 - 1.0]]--
          input_ctx_vis:copy(real_ctx)
          input_mask:copy(real_mask)

          if opt.withInit~=0 then
            local fake_init = netI:forward(input_ctx_vis)
            input_ctx_vis = inpainter.fillIn(input_ctx_vis, input_mask, fake_init)
          end
          local fake
          if opt.noiseGen then
            fake = netG:forward({input_ctx_vis,noise_vis})
          else
            fake = netG:forward(input_ctx_vis)
          end

          --M: Now *fake* is the generated image, so no need to show the CTX with the CENTER.
          --M: but show it with mask.

          -- local maskedout = fake:maskedSelect(input_mask)
          -- print(maskedout:min(), maskedout:max())
          -- print(real_ctx:min(), real_ctx:max())
          -- print(real_full:min(), real_full:max())
          -- real_ctx = real_ctx:cuda()
          -- real_ctx:maskedCopy(input_mask, maskedout)
          if ncin==4 then 
            real_ctx = inpainter.removePadMask(real_ctx, ncin, ncout)
          end
          real_ctx = real_ctx:cuda()
          real_ctx = inpainter.fillIn(real_ctx, input_mask, fake)

          local mididx = math.floor(predLen/2)
          local fake_singleim = fake[{{}, {}, {mididx}, {}, {}}]
          local real_full_singleim = real_full[{{}, {}, {mididx}, {}, {}}]
          local real_ctx_singleim = real_ctx[{{}, {}, {mididx}, {}, {}}]
          disp.image(fake_singleim:view(opt.predLen*opt.batchSize, opt.ncout, fake:size(3), fake:size(4)), {win=opt.display_id, title=opt.name..'f'})
          --disp.image(input_mask:view(opt.predLen*opt.batchSize, opt.nc, fake:size(3), fake:size(4)), {win=opt.display_id * 2, title=opt.name..'m'})
          disp.image(real_full_singleim:view(opt.predLen*opt.batchSize, opt.ncout, fake:size(3), fake:size(4)), {win=opt.display_id * 3, title=opt.name..'r'})
          disp.image(real_ctx_singleim:view(opt.predLen*opt.batchSize, opt.ncout, fake:size(3), fake:size(4)), {win=opt.display_id * 6, title=opt.name..'i'})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G_L2: %.4f   Err_G_GDL: %.4f   Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real, errG_l2 or -1, errG_gdl or -1, 
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   if epoch % 20 == 0 then
      util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG, opt.gpu)
      util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD, opt.gpu)
   end
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
