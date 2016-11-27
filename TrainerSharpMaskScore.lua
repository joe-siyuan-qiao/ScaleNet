--[[----------------------------------------------------------------------------
This trainer only trains the scorebranch of the sharpmask model

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
  self.scoreNet = nn.Sequential():add(model.trunk):add(model.scoreBranch)
  self.criterion = criterion
  self.lr = config.lr
  self.optimState = {
    learningRate = config.lr,
    learningRateDecay = 0,
    momentum = config.momentum,
    dampening = 0,
    weightDecay = config.wd,
  }
  self.ps, self.gs = model.scoreBranch:getParameters()
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()
  self.lossmeter = LossMeter()
  self.scoremeter = BinaryMeter()
  self.modelsv = {model=model:clone('weight', 'bias'),config=config}
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
  local feval = function() return self.criterion.output, self.gs end
  for n, sample in dataloader:run() do
    self:copySamples(sample)
    local model, params = self.scoreNet, self.ps
    local outputs = model:forward(self.inputs)
    local lossbatch = self.criterion:forward(outputs, self.labels)
    model:zeroGradParameters()
    local gradOutputs = self.criterion:backward(outputs, self.labels)
    model:backward(self.inputs, gradOutputs)
    optim.sgd(feval, params, self.optimState)
    self.lossmeter:add(lossbatch)
  end
  -- write log
  local logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
      epoch, timer:time().real/dataloader:size(),self.lossmeter:value())
  print(logepoch)
  io.flush()
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()
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
  self.scoremeter:reset()
  for n, sample in dataloader:run() do
    self:copySamples(sample)
    local outputs = self.scoreNet:forward(self.inputs)
    self.scoremeter:add(outputs, self.labels)
    cutorch.synchronize()
  end
  self.model:training()
  local z, bestmodel = self.scoremeter:value()
  if z > maxacc then
    torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
    maxacc = z
    bestmodel = true
  end

  -- write log
  local logepoch =
    string.format('[test]  | epoch %05d '..
      '| acc %06.2f | bestmodel %s',
      epoch,
      self.scoremeter:value(), bestmodel and '*' or 'x')
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()
  io.flush()

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
      {   1,  40, 1e-3, 5e-4},
      {  41,  80, 5e-4, 5e-4},
      {  81, 1e8, 1e-4, 5e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        self.optimState.learningRate=row[3];
        self.optimState.weightDecay=row[4]
      end
    end
  end
end

return Trainer
