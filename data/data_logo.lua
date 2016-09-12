--[[
    M: This is modified by MKimiSH.
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/data.lua).

    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
]]--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}

local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(n, opt_)
   opt_ = opt_ or {}
   local self = {}
   for k,v in pairs(data) do
      self[k] = v
   end

   local donkey_file = 'donkey_folder_logo.lua'
   if n > 0 then
      local options = opt_
      self.threads = Threads(n,
                             function() require 'torch' end,
                             function(idx)
                                opt = options
                                tid = idx
                                local seed = (opt.manualSeed and opt.manualSeed or 0) + idx
                                torch.manualSeed(seed)
                                torch.setnumthreads(1)
                                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                                assert(options, 'options not found')
                                assert(opt, 'opt not given')
                                print(opt)
                                paths.dofile(donkey_file)
                             end
      )
   else
      if donkey_file then paths.dofile(donkey_file) end
      self.threads = {}
      function self.threads:addjob(f1, f2) f2(f1()) end
      function self.threads:dojob() end
      function self.threads:synchronize() end
   end

   local nSamples = 0
   --M set nSamples variable in all threads.
   self.threads:addjob(function() return trainLoader:size() end,
         function(c) nSamples = c end)
   self.threads:synchronize()
   self._size = nSamples

   self.withMask = false
   --M addjob(f1,f2) -> (f2(f1()))
   if opt_.withMask then
     assert(opt_.maskName and opt_.maskName ~= '', 'You should provide mask name!')
     self.withMask = true
     for i=1, n do
       self.threads:addjob(self._getFromThreadsMask,
                           self._pushResult)
     end
   else
     for i = 1, n do
       self.threads:addjob(self._getFromThreads,
                           self._pushResult)
     end
   end
   return self
end

function data._getFromThreads()
   assert(opt.batchSize, 'opt.batchSize not found')
   --M print('_getFromThreads(%d)', opt.batchSize);
   --M return trainLoader:sample(opt.batchSize)
   return trainLoader:sample2(opt.batchSize)
end

--M sample3 means gt+masked+mask
function data._getFromThreadsMask()
  assert(opt.batchSize, 'opt.batchSize not found')
  return trainLoader:sample3(opt.batchSize)
end

function data._pushResult(...)
   local res = {...}
   if res == nil then
      self.threads:synchronize()
   end
   result[1] = res
   --M the reason of result[1]= rather than result=
   --M is that *result* is a TABLE!!!!
end



function data:getBatch()
  -- queue another job
  if self.withMask then
    self.threads:addjob(self._getFromThreadsMask, self._pushResult)
  else
    self.threads:addjob(self._getFromThreads, self._pushResult)
  end
   self.threads:dojob()
   local res = result[1]
   -- local res_ctx, res_real = result[1], result[2]
   result[1] = nil
   -- result[2] = nil
   if torch.type(res) == 'table' then
      return unpack(res)
      --M after this unpack, the output can be received individually
      --M in original version, 1 output is received
      --M here, 2 outputs are received
   end
   print(type(res))
   return res
end

function data:size()
   return self._size
end

return data
