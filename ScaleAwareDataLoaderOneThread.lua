local M = {}
local DataLoader = torch.class('DataLoader', M)

paths.dofile('ScaleAwareDataSampler.lua')

--------------------------------------------------------------------------------
-- function: create train/val data loaders
function DataLoader.create(config)
  local loaders = {}
  for i, split in ipairs{'train', 'val'} do
    loaders[i] = M.DataLoader(config, split)
  end

  return table.unpack(loaders)
end

--------------------------------------------------------------------------------
-- function: init
function DataLoader:__init(config, split)
  torch.setdefaulttensortype('torch.FloatTensor')
  local seed = config.seed
  torch.manualSeed(seed)
  -- paths.dofile('DataSampler.lua')
  self.ds = DataSampler(config, split)
  local sizes = self.ds:size()
  self.__size = sizes
  self.batch = config.batch
  self.hfreq = config.hfreq
end

--------------------------------------------------------------------------------
-- function: return size of dataset
function DataLoader:size()
  return math.ceil(self.__size / self.batch)
end

--------------------------------------------------------------------------------
-- function: run
function DataLoader:run()
  local size, batch = self.__size, self.batch
  local idx, sample = 1, nil
  local n = 0

  local function customloop()
    if idx > size then return nil end
    local bsz = math.min(batch, size - idx + 1)
    local inputs, labels
    local head
    if torch.uniform() > self.hfreq then head = 1 else head = 2 end
    for i = 1, bsz do
      local input, label = self.ds:get()
      if not inputs then
        local iSz = input:size():totable()
        local mSz = label:size():totable()
        inputs = torch.FloatTensor(bsz, table.unpack(iSz))
        labels = torch.FloatTensor(bsz, table.unpack(mSz))
      end
      inputs[i]:copy(input)
      labels[i]:copy(label)
    end
    idx = idx + batch
    collectgarbage()

    sample = {inputs = inputs, labels = labels, head = head}
    n = n + 1
    return n, sample
  end

  return customloop
end

return M.DataLoader
