require 'image'
local argcheck = require 'argcheck'

local ScaleInfer = torch.class('ScaleInfer')

--------------------------------------------------------------------------------
-- function: init
ScaleInfer.__init = argcheck{
  noordered = true,
  {name="self", type="ScaleInfer"},
  {name="meanstd", type="table"},
  {name="model", type="nn.Container"},
  {name="iSz", type="number", default=160},
  {name="timer", type="boolean", default=false},
  call =
    function(self, meanstd, model, iSz, timer)
      --model
      self.trunk = model.trunk
      self.sHead = model.scaleBranch
      self.net = nn.Sequential():add(model.trunk):add(model.scaleBranch)

      --mean/std
      self.mean, self.std = meanstd.mean, meanstd.std

      -- input size and border width
      self.iSz, self.bw = iSz, iSz/2

      -- timer
      if timer then self.timer = torch.Tensor(6):zero() end

      self.net:replace(function(x)
        if torch.typename(x):find('View') then return nn.Identity()
        else return x end
      end
      )
    end
}

--------------------------------------------------------------------------------
-- function: forward
local inpPad = torch.CudaTensor()
function ScaleInfer:forward(input)
  local inp = input:cuda()
  local h,w = inp:size(2), inp:size(3)
  local outScales = self.net:forward(inp)
  cutorch.synchronize()
  self.outScales = outScales:squeeze():clone()
end

--------------------------------------------------------------------------------
-- function: get top scales
-- arg: sPt - scale probability threshold
-- ret: a table of scales ranked by probabilities
function ScaleInfer:getTopScales(sPt)
  local scales = self.outScales
  scales:exp()
  local resScales = {}
  local _, scaleIdx = torch.sort(scales, true)
  for i = 1, 65 do
    local idx = scaleIdx[i]
    local prob = scales[idx]
    if prob >= sPt then table.insert(resScales, idx) end
  end
  return resScales
end

--------------------------------------------------------------------------------
-- function: get average scale
-- arg: sPt - scale probability threshold
-- ret: average of scales with prob >= sPt
function ScaleInfer:getAvgScale(sPt)
  local scales = self.outScales
  scales:exp()
  local accScale, accTimes = 0.0, 0.0
  for i = 1, 65 do
    local prob = scales[i]
    if prob >= sPt then
      accScale = accScale + prob * i
      accTimes = accTimes + prob
    end
  end
  if accScale > 0 then return {accScale / accTimes}
  else return {31} end
end

--------------------------------------------------------------------------------
-- function: get scales by integral
-- ret: scales selected
function ScaleInfer:getIntScales(sPt, numProp)
  local scales = self.outScales:clone()
  scales:exp()
  scales:pow(0.25)
  sPt = math.pow(sPt, 0.25)
  local resScales, startIdx, endIdx = {}, 1, numProp
  for i = 1, endIdx do table.insert(resScales, -1.0) end
  if scales[29] > sPt then resScales[1] = 29.5; startIdx = 2 end
  if scales[35] > sPt then 
    resScales[endIdx] = 34.5; endIdx = endIdx - 1 
  end
  local pd = torch.Tensor(5000):zero()
  local norm = scales:narrow(1,30,5):sum()
  scales:narrow(1,30,5):div(norm)
  for i = 1, 5 do
    pd:narrow(1,i*1000-999,1000):fill(scales[29+i] / 1000.0)
  end
  local stepsize = 1.0 / (endIdx - startIdx + 2)
  local acc, idx = 0.0, 1
  for i = startIdx, endIdx do
    while acc < stepsize do
      acc = acc + pd[idx]; idx = idx + 1
    end
    acc = 0; resScales[i] = 29.5 + idx / 1000;
  end
  return resScales
end

return ScaleInfer
