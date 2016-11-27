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
cmd:option('-sp', 1, 'the precision of scale estimation')
cmd:option('-inputsize', 1280, 'the maxmimum dimensional size for forwarding')
cmd:option('-alpha', 0.0, 'alpha when blending mu and sigma')
cmd:option('-beta', -0.75, 'beta when blending mean and acc')
cmd:option('-head', 1, '1: seg mask head 2: bbox mask head')
cmd:option('-near', 2, '1: near 0: others')

local config = cmd:parse(arg)
local configblendingalpha = config.alpha
local configblendingbeta = config.beta
local confighead = config.head
local confignear = config.near

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

-- local infer = Infer{
--   np = config.np,
--   scales = scales,
--   meanstd = meanstd,
--   model = model,
--   dm = config.dm,
-- }

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
-- string split utility
-- Reference: http://lua-users.org/wiki/SplitJoin

function stringsplit(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
	 table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

--------------------------------------------------------------------------------
-- start batch operations

-- do the scale estimation
paths.dofile('computeScales.lua')
local scalelistfile = paths.concat('intermediate', 'imgscales.csv')
local scalelist = lines_from(scalelistfile)
collectgarbage()

-- read file
local imglistfile = 'intermediate/imglist.csv'
imglist = lines_from(imglistfile)

local writetofilestring = ''

io.write('[running] ')
io.flush()

for img_idx, img_path in pairs(imglist) do
  -- load image
  local img = image.load(img_path, 3)
  local h,w = img:size(2), img:size(3)

  local maxdim, mindim = math.max(h, w), math.min(h, w)
  local img_scale = 1
  -- h, w = math.floor(h / img_scale), math.floor(w / img_scale)
  -- img = image.scale(img, w, h)

  -- compute the scale pyramid
  local scalesstr, scalesnum = stringsplit(scalelist[img_idx], ','), {}
  for i = 1, #scalesstr do scalesnum[i] = tonumber(scalesstr[i]) end
  local scalestab = {}
  local scalemax, scalemin = 0, 65
  for i = 1, #scalesnum do
    scalemax = math.max(scalemax, scalesnum[i])
    scalemin = math.min(scalemin, scalesnum[i])
  end
  for i = 1, #scalesnum do
    if confignear == 1 then
      local scalesnumtmp = scalesnum[i]
      scalesnum[i] = 2 ^ ((33 - scalesnum[i]) / config.sp) * 192 / maxdim
      if scalesnum[i] * maxdim < 1000 then table.insert(scalestab, scalesnum[i]) end
      scalesnum[i] = 2 ^ ((33 - scalesnumtmp + 0.33) / config.sp) * 192 / maxdim
      if scalesnum[i] * maxdim < 1000 then table.insert(scalestab, scalesnum[i]) end
      scalesnum[i] = 2 ^ ((33 - scalesnumtmp + 0.66) / config.sp) * 192 / maxdim
      if scalesnum[i] * maxdim < 1000 then table.insert(scalestab, scalesnum[i]) end
    end
    if confignear == 0 then
      local scalesnumtmp = scalesnum[i]
      scalesnum[i] = 2 ^ ((33 - scalesnum[i]) / config.sp) * 200 / maxdim
      if scalesnum[i] * maxdim < 1800 then table.insert(scalestab, scalesnum[i]) end
      scalesnum[i] = 2 ^ ((33 - scalesnumtmp - 0.5) / config.sp) * 200 / maxdim
      if scalesnum[i] * maxdim < 1800 then table.insert(scalestab, scalesnum[i]) end
    end
    if confignear == 2 then
      local donothing = true
    end
  end

  for _, scalenumtmp in pairs (scalesnum) do
    local scale = 2 ^ ((33 - scalenumtmp) / config.sp) * 200 / maxdim
    table.insert(scalestab, scale)
  end

  -- print (string.format("ScaleNet+DeepMask is processing %s using %d scales from %.2f to %.2f",
  --   img_path, #scalestab, scalemin, scalemax))
  io.write('. ')
  io.flush()

  local infer = Infer{
    np = config.np,
    scales = scalestab,
    meanstd = meanstd,
    model = model,
    dm = config.dm,
    -- lowmemmode = true,
  }

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
    local widthset = {}
    local yacc, xacc, nacc = 0, 0, 0
    for yidx = 0, h - 1 do
      local yoffset = yidx * w
      local rowwidth = 0
      for xidx = 0, w - 1 do
        if maskstwodimptr[yoffset + xidx] == 1 then
          ymin = mathmin(ymin, yidx)
          xmin = mathmin(xmin, xidx)
          ymax = mathmax(ymax, yidx)
          xmax = mathmax(xmax, xidx)
          rowwidth = rowwidth + 1
          yacc, xacc, nacc = yacc + yidx, xacc + xidx, nacc + 1
        end
      end
      if rowwidth > 0 then table.insert(widthset, rowwidth) end
    end
    if confighead == 2 then
      yacc, xacc = yacc / nacc, xacc / nacc
      local widthtensor = torch.Tensor(widthset)
      local widthmu, widthsigma = widthtensor:mean(), widthtensor:std()
      local hightset = {}
      for xidx = 0, w - 1 do
        local colhight = 0
        for yidx = 0, h - 1 do
          if maskstwodimptr[yidx * w + xidx] == 1 then
            colhight = colhight + 1
          end
        end
        if colhight > 0 then table.insert(hightset, colhight) end
      end
      local highttensor = torch.Tensor(hightset)
      local hightmu, hightsigma = highttensor:mean(), highttensor:std()
      local ymean, xmean = 0.5 * (ymin + ymax), 0.5 * (xmin + xmax)
      ymean = ymean * configblendingbeta + (1 - configblendingbeta) * yacc
      xmean = xmean * configblendingbeta + (1 - configblendingbeta) * xacc
      local bboxwidth = widthmu + configblendingalpha * widthsigma
      local bboxhight = hightmu + configblendingalpha * hightsigma
      ymin, ymax = ymean - 0.5 * bboxhight, ymean + 0.5 * bboxhight
      xmin, xmax = xmean - 0.5 * bboxwidth, xmean + 0.5 * bboxwidth
    end
    writestring[proposal_idx] = img_scale * (ymin + 1) .. ',' .. img_scale *
    (xmin + 1) .. ',' .. img_scale * (ymax + 1) .. ',' .. img_scale *
    (xmax + 1) .. '\n'
  end
  writetofilestring = writetofilestring .. '\n' .. table.concat(writestring)
  infer = nil
  collectgarbage()
end
io.write('\n')
io.flush()

local bboxfilename = paths.concat('intermediate', 'imgbbox.csv')
local bboxfile = io.open(bboxfilename, 'w')
bboxfile:write(writetofilestring)
bboxfile:close()
