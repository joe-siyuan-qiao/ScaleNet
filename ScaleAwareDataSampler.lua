require 'torch'
require 'image'
local tds = require 'tds'
local coco = require 'coco'

local DataSampler = torch.class('DataSampler')

--------------------------------------------------------------------------------
-- function: init
function DataSampler:__init(config,split)
  assert(split == 'train' or split == 'val')

  -- coco api
  local annFile = string.format('%s/annotations/instances_%s2014.json',
  config.datadir,split)
  self.coco = coco.CocoApi(annFile)

  -- mask api
  self.maskApi = coco.MaskApi

  -- mean/std computed from random subset of ImageNet training images
  self.mean, self.std = {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}

  -- class members
  self.datadir = config.datadir
  self.split = split

  self.iSz = config.iSz
  self.objSz = math.ceil(config.iSz*128/224)
  self.wSz = config.iSz + 32
  self.gSz = config.gSz
  self.scale = config.scale
  self.shift = config.shift

  self.imgIds = self.coco:getImgIds()
  self.annIds = self.coco:getAnnIds()
  self.catIds = self.coco:getCatIds()
  self.nImages = self.imgIds:size(1)

  if split == 'train' then self.__size  = config.maxload*config.batch
  elseif split == 'val' then self.__size = config.testmaxload*config.batch end

  if config.hfreq > 0 then
    self.scales = {} -- scale range for score sampling
    for scale = -3,2,.25 do table.insert(self.scales,scale) end
    self:createBBstruct(self.objSz,config.scale)
  end

  collectgarbage()
end
local function log2(x) return math.log(x)/math.log(2) end

--------------------------------------------------------------------------------
-- function: create BB struct of objects for score sampling
-- each key k contain the scale and bb information of all annotations of
-- image k
function DataSampler:createBBstruct(objSz,scale)
  local bbStruct = tds.Vec()

  for i = 1, self.nImages do
    local annIds = self.coco:getAnnIds({imgId=self.imgIds[i]})
    local bbs = {scales = {}}
    if annIds:dim() ~= 0 then
      for i = 1,annIds:size(1) do
        local annId = annIds[i]
        local ann = self.coco:loadAnns(annId)[1]
        local bbGt = ann.bbox
        local x0,y0,w,h = bbGt[1],bbGt[2],bbGt[3],bbGt[4]
        local xc,yc, maxDim = x0+w/2,y0+h/2, math.max(w,h)

        for s = -32,32,1 do
          if maxDim > objSz*2^((s-1)*scale) and
            maxDim <= objSz*2^((s+1)*(scale)) then
            local ss = -s*scale
            local xcS,ycS = xc*2^ss,yc*2^ss
            if not bbs[ss] then
              bbs[ss] = {}; table.insert(bbs.scales,ss)
            end
            table.insert(bbs[ss],{xc,yc,maxDim,category_id=ann.category})
            break
          end
        end
      end
    end
    bbStruct:insert(tds.Hash(bbs))
  end
  collectgarbage()
  self.bbStruct = bbStruct
end

--------------------------------------------------------------------------------
-- function: get size of epoch
function DataSampler:size()
  return self.__size
end

--------------------------------------------------------------------------------
-- function: get a sample
function DataSampler:get()
  local input, label
  repeat
    input, label = self:scaleSamping()
  until label:norm(1) ~= 0

  -- normalize input
  for i=1,3 do input:narrow(1,i,1):add(-self.mean[i]):div(self.std[i]) end

  return input, label
end

--------------------------------------------------------------------------------
-- function: scale sampling
function DataSampler:scaleSamping()
  local iSz, wSz, gSz = self.iSz, self.wSz, self.gSz
  local idx, bb
  repeat
    idx = torch.random(1, self.nImages)
    bb = self.bbStruct[idx]
  until #bb.scales ~= 0

  local imgId = self.imgIds[idx]
  local imgName = self.coco:loadImgs(imgId)[1].file_name
  local pathImg = string.format('%s/%s2014/%s',self.datadir,self.split,imgName)
  local img = image.load(pathImg,3)
  local h,w = img:size(2),img:size(3)
  local maxDim = math.max(h, w)
  local imgResizeRatio = wSz / maxDim
  local h = math.min(math.ceil(h * imgResizeRatio), wSz)
  local w = math.min(math.ceil(w * imgResizeRatio), wSz)
  local img = image.scale(img, w, h)

  -- crop the image into wSz * wSz
  local inp = torch.FloatTensor(3, wSz, wSz)
  inp:fill(0.456)
  local x, y = math.floor((wSz - w) / 2) + 1, math.floor((wSz - h) / 2) + 1
  -- local inp = img:narrow(2, y, wSz):narrow(3, x, wSz)
  inp:narrow(2, y, h):narrow(3, x, w):copy(img)

  -- collect the scales of bboxes of which the centers are within the cropped
  local lbl = torch.Tensor(65)
  lbl:fill(0)
  for scaleIdx = 1, #bb.scales do
    local scale = bb.scales[scaleIdx]
    local s = 2^-scale * imgResizeRatio
    for bbIdx = 1, #bb[scale] do
      local bbX, bbY = bb[scale][bbIdx][1], bb[scale][bbIdx][2]
      local bbDim = bb[scale][bbIdx][3]
      bbX, bbY = bbX * s, bbY * s
      bbX, bbY = bbX - x + 1, bbY - y + 1
      if 1 == 1 then
        local bbScale = bbDim * imgResizeRatio / self.objSz
        local logScale = log2(bbScale)
        local logScaleIdxFloat = logScale * 1 + 33
        local logScaleIdxCeil = math.ceil(logScaleIdxFloat)
        local logScaleIdxFloor = math.floor(logScaleIdxFloat)
        local logScaleIdxFloatCeil = logScaleIdxFloat - logScaleIdxFloor
        local logScaleIdxFloatFloor = logScaleIdxCeil - logScaleIdxFloat
        lbl[logScaleIdxFloor]=lbl[logScaleIdxFloor]+logScaleIdxFloatFloor
        lbl[logScaleIdxCeil]=lbl[logScaleIdxCeil]+logScaleIdxFloatCeil
      end
    end
  end
  local lblOneNorm = lbl:norm(1)
  if lblOneNorm ~= 0 then lbl = lbl:div(lblOneNorm) end
  return inp, lbl
end

return DataSampler
