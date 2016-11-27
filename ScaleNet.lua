--[[----------------------------------------------------------------------------
When initialized, it creates/load the common trunk and the scale estimation
ScaleNet class members:
  - trunk: the common trunk (modified pre-trained resnet50)
  - scaleBranch: the scale estimation architecture
------------------------------------------------------------------------------]]

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
if not nn.SpatialSymmetricPadding then
  paths.dofile('SpatialSymmetricPadding.lua')
end
local utils = paths.dofile('modelUtils.lua')

local ScaleNet, _ = torch.class('nn.ScaleNet', 'nn.Container')

--------------------------------------------------------------------------------
-- function: constructor
function ScaleNet:__init(config)
  self.batch = config.batch

  -- create resnet50 trunk
  self:createTrunk(config)

  -- create scale head
  self:createScaleBranch(config)

  -- number of parameters
  local npt, nps = 0, 0
  local p1, p2 = self.trunk:parameters(), self.scaleBranch:parameters()
  for k, v in pairs(p1) do npt = npt + v:nElement() end
  for k, v in pairs(p2) do nps = nps + v:nElement() end
  print(string.format('| number of parameters trunk: %d', npt))
  print(string.format('| number of parameters scale branch: %d', nps))
  print(string.format('| number of parameters total: %d', npt + nps))
end

--------------------------------------------------------------------------------
-- function: create common trunk
function ScaleNet:createTrunk(config)
  -- size of feature maps at end of trunk
  self.fSz = config.iSz/16

  -- load trunk
  local trunk = torch.load('pretrained/resnet-50.t7')

  -- remove BN
  -- utils.BNtoFixed(trunk, true)

  -- remove fully connected layers
  trunk:remove();trunk:remove();trunk:remove();trunk:remove()

  -- crop central pad
  trunk:add(nn.SpatialZeroPadding(-1,-1,-1,-1))

  -- -- add common extra layers
  trunk:add(cudnn.SpatialConvolution(1024,4096,1,1,1,1))
  trunk:add(cudnn.SpatialBatchNormalization(4096))
  trunk:add(cudnn.ReLU())

  -- from scratch? reset the parameters
  if config.scratch then
    for k,m in pairs(trunk.modules) do if m.weight then m:reset() end end
  end

  -- symmetricPadding
  utils.updatePadding(trunk, nn.SpatialSymmetricPadding)

  self.trunk = trunk:cuda()
  return trunk
end

--------------------------------------------------------------------------------
-- function: create scale branch
function ScaleNet:createScaleBranch(config)
  local scaleBranch = nn.Sequential()
  scaleBranch:add(cudnn.SpatialConvolution(4096,4096,1,1,1,1))
  scaleBranch:add(cudnn.SpatialBatchNormalization(4096))
  scaleBranch:add(cudnn.ReLU())
  scaleBranch:add(cudnn.SpatialConvolution(4096,65,1,1,1,1))
  -- scaleBranch:add(cudnn.SpatialBatchNormalization(65))
  -- scaleBranch:add(cudnn.ReLU())
  scaleBranch:add(cudnn.SpatialAveragePooling(self.fSz, self.fSz, 1, 1))
  scaleBranch:add(nn.View(self.batch, 65))
  scaleBranch:add(nn.LogSoftMax())

  self.scaleBranch = scaleBranch:cuda()
  return self.scaleBranch
end

--------------------------------------------------------------------------------
-- function: training
function ScaleNet:training()
  self.trunk:training(); self.scaleBranch:training()
end

--------------------------------------------------------------------------------
-- function: evaluate
function ScaleNet:evaluate()
  self.trunk:evaluate(); self.scaleBranch:evaluate()
end

--------------------------------------------------------------------------------
-- function: to cuda
function ScaleNet:cuda()
  self.trunk:cuda(); self.scaleBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: to float
function ScaleNet:float()
  self.trunk:float(); self.scaleBranch:float()
end

--------------------------------------------------------------------------------
-- function: inference (used for full scene inference)
function ScaleNet:inference()
  self.trunk:evaluate()
  self.scaleBranch:evaluate()

  -- utils.linear2convTrunk(self.trunk,self.fSz)
  -- utils.linear2convHead(self.scaleBranch)

  self:cuda()
end

--------------------------------------------------------------------------------
-- function: clone
function ScaleNet:clone(...)
  local f = torch.MemoryFile("rw"):binary()
  f:writeObject(self); f:seek(1)
  local clone = f:readObject(); f:close()

  if select('#',...) > 0 then
    clone.trunk:share(self.trunk,...)
    clone.scaleBranch:share(self.scaleBranch,...)
  end

  return clone
end

return nn.ScaleNet
