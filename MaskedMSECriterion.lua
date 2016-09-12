require 'nn'
require 'nngraph'

maskMSE, mseparent = torch.class('nn.MaskedMSECriterion', 'nn.Criterion')
-- torch.setdefaulttensortype('torch.FloatTensor')
-- mWeight=maskWeight
function maskMSE:__init(mWeight)
  mseparent:__init(self)
  self.mWeight = mWeight or 1

  local X = nn.Identity()()
  local Xhat = nn.Identity()()
  local M = nn.Identity()()

  local wM = nn.AddConstant(mWeight, true)(nn.MulConstant(1-mWeight)(M))
  local MSE = nn.Square()(nn.CSubTable(){X, Xhat})
  local wMSE = nn.CMulTable(){wM, MSE}

  self.net = nn.gModule({X, Xhat, M}, {wMSE})
  self.crit = nn.AbsCriterion()
  self.target1 = torch.Tensor()
end

function maskMSE:setMask(m)
  assert(m:type() == 'torch.ByteTensor')
  self.mask = m:double()
end

function maskMSE:updateOutput(input, target)
  self.netoutput = self.net:updateOutput{input, target, self.mask}
 -- 0 print(self.netoutput[2])
  self.target1:resizeAs(self.netoutput):zero()
  self.loss = self.crit:updateOutput(self.netoutput, self.target1)
  return self.loss
end

function maskMSE:updateGradInput(input, target)
  local gradInput = self.crit:updateGradInput(self.netoutput, self.target1)
  self.gradInput =
    self.net:updateGradInput({input, target, self.mask}, gradInput)[1]
  return self.gradInput
end