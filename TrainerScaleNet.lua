--[[----------------------------------------------------------------------------
Training and testing loop for ScaleNet
------------------------------------------------------------------------------]]

local optim = require 'optim'
paths.dofile('trainMeters.lua')

local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, criterion, config)
  -- training params
  self.config = config
  self.model = model
  self.scaleNet = nn.Sequential():add(model.trunk):add(model.scaleBranch)
  self.criterion = criterion
  self.lr = config.lr
  self.optimState = {}
  for k,v in pairs({'trunk', 'scale'}) do
    self.optimState[v] = {
      learningRate = config.lr,
      learningRateDecay = 0,
      momentum = config.momentum,
      dampening = 0,
      weightDecay = config.wd,
    }
  end

  -- params and gradparams
  self.pt, self.gt = model.trunk:getParameters()
  self.ps, self.gs = model.scaleBranch:getParameters()

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter = LossMeter()

  -- log
  self.modelsv = {model=model:clone('weight', 'bias', 'running_mean',
    'running_var'), config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
end

--------------------------------------------------------------------------------
-- function: train
function Trainer:train(epoch, dataloader)
  self.model:training()
  self:updateScheduler(epoch)
  self.lossmeter:reset()

  local timer = torch.Timer()

  local fevaltrunk = function() return self.model.trunk.output, self.gt end
  local fevalscale = function() return self.criterion.output, self.gs end

  local collectgarbagecount = 0
  for n, sample in dataloader:run() do
    -- copy samples to the GPU
    self:copySamples(sample)

    -- forward/backward
    local model, params, feval, optimState
    model, params = self.scaleNet, self.ps
    feval, optimState = fevalscale, self.optimState.scale

    local outputs= model:forward(self.inputs)
    local lossbatch = self.criterion:forward(outputs, self.labels)
    model:zeroGradParameters()
    local gradOutputs = self.criterion:backward(outputs, self.labels)
    gradOutputs:mul(self.inputs:size(1))
    model:backward(self.inputs, gradOutputs)

    -- optimize
    optim.sgd(fevaltrunk, self.pt, self.optimState.trunk)
    optim.sgd(feval, params, optimState)

    -- update loss
    self.lossmeter:add(lossbatch)

    collectgarbagecount = collectgarbagecount + 1
    if collectgarbagecount == 100 then
      collectgarbagecount = 0
      collectgarbage()
    end
  end

  -- write log
  local logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
      epoch, timer:time().real/dataloader:size(),self.lossmeter:value())
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  if epoch%50 == 0 then
    torch.save(string.format('%s/model_%d.t7', self.rundir, epoch),
      self.modelsv)
  end

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: test
local maxacc = 0
function Trainer:test(epoch, dataloader)
  self.model:evaluate()
  local iouSum, numRecord = 0.0, 0
  for n, sample in dataloader:run() do
    self:copySamples(sample)
    local outputs = self.scaleNet:forward(self.inputs)
    cutorch.synchronize()
    local resizedOutputs = outputs:view(self.labels:size())
    for i = 1, self.labels:size(1) do
      numRecord = numRecord + 1
      local label = self.labels[i]
      local pred = resizedOutputs[i]
      local unioncnt = 0
      local intercnt = 0
      for j = 1, 65 do
        local haslabel = label[j] > 0
        local haspred = pred[j] > -4.6051701859881 -- log(0.01)
        if haslabel and haspred then intercnt = intercnt + 1 end
        if haslabel or haspred then unioncnt = unioncnt + 1 end
      end
      iouSum = iouSum + intercnt / unioncnt
    end
  end
  self.model:training()

  -- check if bestmodel so far
  local z,bestmodel = iouSum / numRecord
  if z > maxacc then
    torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
    maxacc = z
    bestmodel = true
  end

  -- write log
  local logepoch =
    string.format('[test]  | epoch %05d '..
      '| mean iou: %.6f | bestmodel %s',
      epoch, z, bestmodel and '*' or 'x')
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: copy inputs/labels to CUDA tensor
function Trainer:copySamples(sample)
  self.inputs:resize(sample.inputs:size()):copy(sample.inputs)
  self.labels:resize(sample.labels:size()):copy(sample.labels)
end

--------------------------------------------------------------------------------
-- function: update training schedule according to epoch
function Trainer:updateScheduler(epoch)
  if self.lr == 0 then
    local regimes = {
      {   1,  50, 1e-3, 5e-4},
      {  51, 120, 5e-4, 5e-4},
      { 121, 175, 1e-4, 5e-4},
      { 176, 1e8, 1e-5, 5e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        for k,v in pairs(self.optimState) do
          v.learningRate=row[3]; v.weightDecay=row[4]
        end
      end
    end
  end
end

return Trainer
