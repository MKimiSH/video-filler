require 'image'

local inpaint_utils = {}

local function dim_hw(T)
  assert(T:dim()==3 or T:dim()==4)
  if T:dim()==3 then
    return 2, 3
  else
    return 3, 4
  end
end

-- complaint: why image.scale asks for w before h........
local function scale_with4D(T, w, h)
  assert(T:dim()==3 or T:dim()==4)
  if T:dim()==3 then
    return image.scale(T, w, h)
  else
    local l=T:size(1)
    for i=1,l do
      T[i] = image.scale(T[i], w, h)
    end
  end
  return T
end

-- should deal with 3D or 4D (with batch) input
-- dst:size() should not change, if withScale==true then mask should be rescaled.
function inpaint_utils.maskOut(dst, mask, mValue, wScale)
  local withScale = wScale or false
  local dimh, dimw = dim_hw(dst)
  local mdimh, mdimw = dim_hw(mask)
  assert(mdimh<=dimh, 'What a wierd mask?!!')
  if not withScale then
    assert(dst:size(dimh)==mask:size(mdimh))
    assert(dst:size(dimw)==mask:size(mdimw))
  else -- scale mask
    -- I don't expect 4D mask.... but it's there!
    mask = image.scale(mask, dst:size(dimw), dst:size(dimh))
  end

  if dst:dim() == 4 then
    if(mask:dim() == 4) then
      dst:maskedFill(mask, mValue)
      return dst
    end
    assert(mask:size(1) == dst:size(2), 'Mask should be the same size as a single batch in batch mode')--..mask:size()..dst:size())
    for b=1,dst:size(1) do
      dst[b]:maskedFill(mask, mValue)
    end
  else -- 3D dst
    assert(dst:size(1) % mask:size(1) == 0, 'Mask length should be divisible by dst length in non-batch mode')
    local step = mask:size(1)
    for b=1, dst:size(1), step do
      dst[{{b,b+step-1},{},{}}]:maskedFill(mask, mValue)
    end
  end
  return dst
end

-- if withScale==true then mask and src should be both rescaled.
function inpaint_utils.fillIn(dst, mask, src, wScale)
  local withScale = wScale or false
  assert(dst:dim() == src:dim())
  local dimh, dimw = dim_hw(dst)
  assert(dst:size(dimh-1) == src:size(dimh-1), 'src and dst should contain same number of images')
  local mdimh, mdimw = dim_hw(mask)
  assert(mdimh<=dimh, 'What a wierd mask?!!')

  if not withScale then
    assert(dst:size(dimh) == src:size(dimh) and dst:size(dimh) == mask:size(mdimh))
    assert(dst:size(dimw) == src:size(dimw) and dst:size(dimw) == mask:size(mdimw))
  else
    mask = scale_with4D(mask, dst:size(dimw), dst:size(dimh))
    src = scale_with4D(src, dst:size(dimw), dst:size(dimh))
  end

  if dst:dim() == 4 then
    if mask:dim() == 4 then
      local masked = src:maskedSelect(mask)
      dst:maskedCopy(mask, masked)
      return dst
    end
    assert(mask:size(1) == dst:size(2), 'Mask should be the same size as a single batch in batch mode')--..mask:size()..dst:size())
    for b=1,dst:size(1) do
      -- dst[b]:maskedFill(mask, mValue)
      local masked = src[b]:maskedSelect(mask)
      dst[b]:maskedCopy(mask, masked)
    end
  else -- 3D dst
    assert(dst:size(1) % mask:size(1) == 0, '#mask should be divisible by #dst in non-batch mode')
    local step = mask:size(1)
    for b=1, dst:size(1), step do
      local masked = src[{{b, b+step-1}, {}, {}}]:maskedSelect(mask)
      dst[{{b,b+step-1},{},{}}]:maskedCopy(mask, masked)
      -- dst[{{b,b+step-1},{},{}}]:maskedFill(mask, mValue)
    end
  end
  return dst
end

local function removePadMask3D(dst, ncin, ncout)
  --print(dst:size())
  assert(dst:size(1)%ncin == 0, "ndims not matching")
  
  local sizes = torch.LongStorage({dst:size(1)*ncout/ncin, dst:size(2), dst:size(3)})
  local out = torch.Tensor(sizes)
  --print(out:size())
  for i=1, dst:size(1), ncin do
    local outi = (i-1)*ncout/ncin + 1
    --print(outi)
    out[{{outi, outi+ncout-1}, {}, {}}] = dst[{{i, i+ncout-1}, {}, {}}]:clone()
  end
  return out
end

-- change dst from ncin to ncout channels
-- first dim must be channels!!!
function inpaint_utils.removePadMask(dst, ncin, ncout)
  --print(dst:size())
  assert(dst:dim()==3 or dst:dim()==4, "first dim must be channels or batchSize!")
  -- assert(dst:size(1)%ncin == 0, "ndims not matching")
  if dst:dim()==3 then 
    return removePadMask3D(dst, ncin, ncout)
  else
    local sizes = dst:size()
    sizes[2] = sizes[2]*ncout/ncin
    local out = torch.Tensor(sizes)
    for i=1,sizes[1] do
      local outi = removePadMask3D(dst[i], ncin, ncout)
      out[i] = outi:clone()
    end
    return out
  end
  --return out
end


return inpaint_utils