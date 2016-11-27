--[[----------------------------------------------------------------------------
This is the interface for python to call. This is sopposed to do a batch mask
extraction procedures.

Reference: computeProposals.lua
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'image'
require 'math'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-np', 1000, 'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', false, 'use DeepMask')
cmd:option('-nw', 8, 'the number of the workers')
cmd:option('-inputsize', 640, 'the maxmimum dimensional size for forwarding')

local config = cmd:parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load moodel
paths.dofile('DeepMask.lua')
if config.dm == false then
  paths.dofile('SharpMask.lua')
end

if config.dm == false then
  m = torch.load('pretrained/sharpmask/model.t7')
else
  m = torch.load('pretrained/deepmask/model.t7')
end
local model = m.model
model:inference(config.np)
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end

if config.dm == false then
  paths.dofile('InferSharpMask.lua')
else
  paths.dofile('InferDeepMask.lua')
end

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = model,
  dm = config.dm,
  -- lowmemmode = true,
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

-- -- tests the functions above
-- local file = '.gitignore'
-- local lines = lines_from(file)
--
-- -- print all line numbers and their contents
-- for k,v in pairs(lines) do
-- print('line[' .. k .. ']', v)
-- end

--------------------------------------------------------------------------------
-- start batch operations

-- read file
local imglistfile = 'intermediate/imglist.csv'
imglist = lines_from(imglistfile)

local writetofilestring = ''

for img_idx, img_path in pairs(imglist) do
  -- load image
  io.write(string.format('DeepMask is processing %s\n', img_path))
  io.flush()
  local img = image.load(img_path, 3)
  local h,w = img:size(2), img:size(3)

  local maxdim = math.max(h, w)
  local img_scale = maxdim / config.inputsize
  img_scale = 1.0
  h, w = math.floor(h / img_scale), math.floor(w / img_scale)
  img = image.scale(img, w, h)

  -- forward all scales
  infer:forward(img)

  -- get top proposals
  local masks, scores = infer:getTopProps(.2, h, w)

  local writestring = {}
  local proposal_num = masks:size(1)
  for proposal_idx = 1, proposal_num do
    writestring[proposal_idx] = ''
  end
  for proposal_idx = 1, proposal_num do
    local maskstwodim = masks[proposal_idx]
    local h, w = maskstwodim:size(1), maskstwodim:size(2)
    local maskstwodimptr = maskstwodim:data()
    local ymin, ymax, xmin, xmax = h - 1, 0, w - 1, 0
    local mathmin, mathmax = math.min, math.max
    for yidx = 0, h - 1 do
      local yoffset = yidx * w
      for xidx = 0, w - 1 do
        if maskstwodimptr[yoffset + xidx] == 1 then
          ymin = mathmin(ymin, yidx)
          xmin = mathmin(xmin, xidx)
          ymax = mathmax(ymax, yidx)
          xmax = mathmax(xmax, xidx)
        end
      end
    end
    writestring[proposal_idx] = img_scale * (ymin + 1) .. ',' .. img_scale *
    (xmin + 1) .. ',' .. img_scale * (ymax + 1) .. ',' .. img_scale *
    (xmax + 1) .. '\n'
  end
  writetofilestring = writetofilestring .. '\n' .. table.concat(writestring)
end

local bboxfilename = paths.concat('intermediate', 'imgbbox.csv')
local bboxfile = io.open(bboxfilename, 'w')
bboxfile:write(writetofilestring)
bboxfile:close()
