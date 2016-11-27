local cjson = require 'cjson'
local tds = require 'tds'
local coco = require 'coco'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-datadir', 'data/', 'data directory')
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-split', 'val', 'dataset split to be used (train/val)')
cmd:option('-np', 1000,'number of proposals')
cmd:option('-thr', .2, 'mask binary threshold')
cmd:option('-startAt', 1, 'start image id')
cmd:option('-endAt', 5000, 'end image id')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- get list of eval images
local annFile = string.format('%s/annotations/instances_%s2014.json',
  config.datadir,config.split)
local coco = coco.CocoApi(annFile)
local imgIds = coco:getImgIds()
imgIds,_ = imgIds:sort()

--------------------------------------------------------------------------------
-- function: get image path
local function getImgPath(datadir, split, filename)
  local pathImg = string.format('%s/%s2014/%s',datadir,split,filename)
  return pathImg
end

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
-- run
print('| start eval'); io.stdout:flush()
local recall1k, recall1h, recall10, numOfGt = {}, {}, {}, 0
for i = 0, 9 do
  recall1k[i / 20 + 0.5] = 0; recall1h[i / 20 + 0.5] = 0
  recall10[i / 20 + 0.5] = 0
end
for k = config.startAt, config.endAt / 20 do
  local imgPathString = ''
  local annsTable = {}
  for kk = 1, 20 do
    local imgId = imgIds[k * 20 - 20 + kk]
    local img = coco:loadImgs(imgId)[1]
    local annIds = coco:getAnnIds({imgId=imgId})
    local anns = coco:loadAnns(annIds)
    table.insert(annsTable, anns)
    local imgPath = getImgPath(config.datadir, config.split, img.file_name)
    imgPathString = imgPathString .. imgPath .. '\n'
  end
  local imgPathFileName = paths.concat('intermediate', 'imglist.csv')
  local imgPathFile = io.open(imgPathFileName, 'w')
  imgPathFile:write(imgPathString)
  imgPathFile:close()
  os.execute(string.format('th ScaleAwarePythonInterface.lua'))
  local bboxfilename = paths.concat('intermediate', 'imgbbox.csv')
  local bboxlines = lines_from(bboxfilename)
  local bboxTable = {}
  for _, line in pairs(bboxlines) do
    if line == '' then
      bboxTable[#bboxTable + 1] = {}
    else
      local bboxstr = stringsplit(line, ',')
      local bboxentry = {}
      for _, bbox in pairs(bboxstr) do
        table.insert(bboxentry, tonumber(bbox))
      end
      table.insert(bboxTable[#bboxTable], bboxentry)
    end
  end
  assert(#bboxTable == 20 and #annsTable == 20)
  for kk = 1, 20 do
    local anns, bboxs = annsTable[kk], bboxTable[kk]
    for annidx = 1, #anns do
      local gt = anns[annidx].bbox
      local xl, yh, xr, yl = gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]
      local maxiou1k, maxiou1h, maxiou10 = -1.0, -1.0, -1.0
      for bboxidx, bbox in pairs(bboxs) do
        local pyh, pxl, pyl, pxr = bbox[1], bbox[2], bbox[3], bbox[4]
        pyh, pxl, pyl, pxr = pyh - 1, pxl - 1, pyl - 1, pxr - 1
        local iyh, iyl = math.max(pyh, yh), math.min(yl, pyl)
        local ixl, ixr = math.max(pxl, xl), math.min(xr, pxr)
        local iarea = (iyl - iyh + 1) * (ixr - ixl + 1)
        local parea = (pyl - pyh + 1) * (pxr - pxl + 1)
        local garea = (yl - yh + 1) * (xr - xl + 1)
        local iou = (iarea) / (parea + garea - iarea)
        if iyl >= iyh and ixr >= ixl and iou > maxiou1k then 
          maxiou1k = iou 
        end
        if bboxidx <= 100 and iyl >= iyh and ixr >= ixl 
            and iou > maxiou1h then maxiou1h = iou end
        if bboxidx <= 10 and iyl >= iyh and ixr >= ixl
            and iou > maxiou10 then maxiou10 = iou end
      end
      for i = 0, 9 do
        local threshold = i / 20 + 0.5
        if maxiou1k >= threshold then
          recall1k[threshold] = recall1k[threshold] + 1
        end
        if maxiou1h >= threshold then
          recall1h[threshold] = recall1h[threshold] + 1
        end
        if maxiou10 >= threshold then
          recall10[threshold] = recall10[threshold] + 1
        end
      end
      numOfGt = numOfGt + 1
    end
  end
  local recallAcc1k, recallAcc1h, recallAcc10 = 0.0, 0.0, 0.0
  for i = 0, 9 do
    recallAcc1k = recallAcc1k + recall1k[i / 20 + 0.5] / numOfGt
    recallAcc1h = recallAcc1h + recall1h[i / 20 + 0.5] / numOfGt
    recallAcc10 = recallAcc10 + recall10[i / 20 + 0.5] / numOfGt
  end
  print (string.format("[testing] | %08d | %.6f@10 | %.6f@1h | %.6f@1k",
    numOfGt, recallAcc10 / 10, recallAcc1h / 10, recallAcc1k / 10))
  io.stdout:flush()
end
