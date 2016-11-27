--[[----------------------------------------------------------------------------
Training and testing loop for ScaleNet
------------------------------------------------------------------------------]]

local optim = require 'optim'
paths.dofile('trainMeters.lua')

local Tester = torch.class('Tester')

--------------------------------------------------------------------------------
-- function: init
function Tester:__init(model, criterion, config)
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
  self.modelsv = {model=model:clone('weight', 'bias'), config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()

  -- test log probability threshold
  self.logprobthres = config.logprobthres
end

--------------------------------------------------------------------------------
-- function: test
function Tester:test(epoch, dataloader)
  self.model:evaluate()
  local logprobthres = self.logprobthres
  local numRecord = 0
  local iouSum = {}
  local recallSum = {}
  for i = 1, #logprobthres do iouSum[i] = 0.0; recallSum[i] = 0.0 end
  for n, sample in dataloader:run() do
    self:copySamples(sample)
    local outputs = self.scaleNet:forward(self.inputs)
    cutorch.synchronize()
    local resizedOutputs = outputs:view(self.labels:size())
    for tidx = 1, #logprobthres do
      local logprob = math.log(logprobthres[tidx])
      for i = 1, self.labels:size(1) do
        if tidx == 1 then numRecord = numRecord + 1 end
        local label = self.labels[i]
        local pred = resizedOutputs[i]
        local unioncnt = 0
        local intercnt = 0
        local gtcnt = 0
        for j = 1, 65 do
          local haslabel = label[j] > 0
          local haspred = pred[j] > logprob
          if haslabel and haspred then intercnt = intercnt + 1 end
          if haslabel or haspred then unioncnt = unioncnt + 1 end
          if haslabel then gtcnt = gtcnt + 1 end
        end
        iouSum[tidx] = iouSum[tidx] + intercnt / unioncnt
        recallSum[tidx] = recallSum[tidx] + intercnt / gtcnt
      end
    end
  end

  -- write log
  local logepoch = '[test]\n'
  for i = 1, #logprobthres do
    logepoch = logepoch .. string.format('| threshold: %.3f | mean iou: ' ..
      '%.5f | mean recall: %.5f\n', logprobthres[i], iouSum[i] / numRecord,
      recallSum[i] / numRecord)
  end
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: copy inputs/labels to CUDA tensor
function Tester:copySamples(sample)
  self.inputs:resize(sample.inputs:size()):copy(sample.inputs)
  self.labels:resize(sample.labels:size()):copy(sample.labels)
end

--------------------------------------------------------------------------------
-- function: update training schedule according to epoch
function Tester:updateScheduler(epoch)
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

return Tester
