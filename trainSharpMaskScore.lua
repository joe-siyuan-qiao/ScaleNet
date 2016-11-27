require 'torch'
require 'cutorch'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('train DeepMask or SharpMask')
cmd:text()
cmd:text('Options:')
cmd:option('-rundir', 'exps/', 'experiments directory')
cmd:option('-datadir', 'data/', 'data directory')
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-nthreads', 4, 'number of threads for DataSampler')
cmd:option('-reload', '', 'reload a network from given directory')
cmd:text()
cmd:text('Training Options:')
cmd:option('-batch', 32, 'training batch size')
cmd:option('-lr', 0, 'learning rate (0 uses default lr schedule)')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-wd', 5e-4, 'weight decay')
cmd:option('-maxload', 4000, 'max number of training batches per epoch')
cmd:option('-testmaxload', 500, 'max number of testing batches')
cmd:option('-maxepoch', 300, 'max number of training epochs')
cmd:option('-iSz', 160, 'input size')
cmd:option('-oSz', 56, 'output size')
cmd:option('-gSz', 112, 'ground truth size')
cmd:option('-shift', 16, 'shift jitter allowed')
cmd:option('-scale', .25, 'scale jitter allowed')
cmd:option('-hfreq', 0.5, 'mask/score head sampling frequency')
cmd:option('-scratch', false, 'train DeepMask with randomly initialize weights')
cmd:option('-reloadepoch', 1, 'the starting epoch for reloading')

local config = cmd:parse(arg)
configreloadepoch = config.reloadepoch or 1
confignthreads = config.nthreads or 2

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)
math.randomseed(config.seed)

config.hfreq = 1.1 -- train only head head

paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

--------------------------------------------------------------------------------
-- reload the pre-trained sharpmask
local epoch, model
local reloadpath = config.reload
local m = torch.load(string.format('%s/model.t7', config.reload))
model = m.model
-- config.reload = reloadpath
config.nthreads = confignthreads

--------------------------------------------------------------------------------
-- directory to save log and model
local pathsv = 'scorebranch/exp'
config.rundir = cmd:string(
  paths.concat(config.rundir, pathsv),
  config,{rundir=true, gpu=true, reload=true, datadir=true, dm=true} --ignore
)

print(string.format('| running in directory %s', config.rundir))
os.execute(string.format('mkdir -p %s',config.rundir))

--------------------------------------------------------------------------------
-- network and criterion
model = model or (trainSm and nn.SharpMask(config) or nn.DeepMask(config))
local criterion = nn.SoftMarginCriterion():cuda()

--------------------------------------------------------------------------------
-- initialize data loader
local DataLoader = paths.dofile('DataLoader.lua')
local trainLoader, valLoader = DataLoader.create(config)

--------------------------------------------------------------------------------
-- initialize Trainer (handles training/testing loop)
paths.dofile('TrainerSharpMaskScore.lua')
local trainer = Trainer(model, criterion, config)

--------------------------------------------------------------------------------
-- do it
epoch = configreloadepoch
print('| start training')
for i = 1, config.maxepoch do
  if i == 1 then print ('Start training from epoch ' .. epoch) end
  if i == 1 then trainer:test(epoch, valLoader) end
  trainer:train(epoch,trainLoader)
  if i%2 == 0 then trainer:test(epoch,valLoader) end
  epoch = epoch + 1
end
