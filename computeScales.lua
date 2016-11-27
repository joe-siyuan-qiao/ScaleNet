require 'torch'
require 'cutorch'
require 'image'
require 'math'

--------------------------------------------------------------------------------
-- initialize
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}
-- local meanstd = {mean = { 0.0, 0.0, 0.0 }, std = { 1.0, 1.0, 1.0 }}


--------------------------------------------------------------------------------
-- load moodel
paths.dofile('ScaleNet.lua')
local m = torch.load('pretrained/scalenet/model.t7')
local model = m.model
model:inference()
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
paths.dofile('InferScaleNet.lua')

local infer = ScaleInfer{
  meanstd = meanstd,
  model = model,
}

--------------------------------------------------------------------------------
-- file read and write utility
-- Reference: http://stackoverflow.com/questions/
-- 11201262/how-to-read-data-from-a-file-in-lua

-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end

--------------------------------------------------------------------------------
-- start batch operations

-- read file
local imglistfile = 'intermediate/imglist.csv'
imglist = lines_from(imglistfile)

local writetofilestring = ''

for img_idx, img_path in pairs(imglist) do
  -- load image
  local img = image.load(img_path, 3)
  local h,w = img:size(2), img:size(3)

  local wSz = 192
  local maxDim = math.max(h, w)
  local imgResizeRatio = wSz / maxDim
  local h = math.min(math.ceil(h * imgResizeRatio), wSz)
  local w = math.min(math.ceil(w * imgResizeRatio), wSz)
  local img = image.scale(img, w, h)

  -- crop the image into wSz * wSz
  local inp = torch.FloatTensor(1, 3, wSz, wSz)
  inp:fill(0.456)
  local x, y = math.floor((wSz - w) / 2) + 1, math.floor((wSz - h) / 2) + 1
  -- local inp = img:narrow(2, y, wSz):narrow(3, x, wSz)
  inp:narrow(3, y, h):narrow(4, x, w):copy(img)

  -- forward the image
  infer:forward(inp)

  -- get top scales
  local scales = infer:getIntScales(0.05, 10)
  local writestring = {}
  for i = 1, #scales do
    table.insert(writestring, string.format('%f,', scales[i]))
  end
  writetofilestring = writetofilestring .. table.concat(writestring) .. '\n'
end

local scalefilename = paths.concat('intermediate', 'imgscales.csv')
local scalefile = io.open(scalefilename, 'w')
scalefile:write(writetofilestring)
scalefile:close()
