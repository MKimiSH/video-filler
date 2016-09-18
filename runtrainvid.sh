# This script uses train_vid_weighted.lua to train (temporarily no finetune)
# possible parameters:
# batchSize = 16
# predLen = 4
# loadSize = 350
# fineSize = 128
# nBottleneck = 4000
# nef = 64, ngf = 64, ndf = 64
# nc = 3
# wtl2 = 0.999
# weight_nomask = 0.05
# wtgdl = 0
# overlapPred = 0 and should be kept 0
# nThreads = 1
# niter = 500
# ntrain = 25600
# lr = 0.0002 learning rate
# beta1 = 0.5 
# display = 1, display_id = 10, display_iter = 50, display_port = 8000
# gpu = 0
# name = 'train1'
# manualSeed = 0
# maskName = 'maskppp.png'  -- should use maskppp.png because mask.png cannot cover whole logo
# maskValue = 110/255
# loadName = ''
# loadIter = 0

######################################## FOLLOWING IS BASH ##########################################

# changemaskvalue
# DATA_ROOT=dataset/train nBottleneck=4000 name=changemaskvalue loadSize=360 gpu=2 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=1 maskValue=0 th train_vid_weighted.lua

# test fine-tune
# DATA_ROOT=dataset/train nBottleneck=4000 name=testfinetune loadIter=40 loadName=changemaskvalue/changemaskvalue loadSize=360 gpu=2 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=1 th train_vid_weighted.lua

### 0906 ###
# test new dataset - 0906
# DATA_ROOT=dataset/train nBottleneck=5000 name=testgt1 loadSize=280 gpu=2 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=1 th train_vid_weighted.lua

# test small loadSize and only l2+gdl and tune weight_nomask -0906
# not working...
# !!Diverges: DATA_ROOT=dataset/train nBottleneck=5000 name=testl2gdl loadSize=240 gpu=2 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=0.05 wtl2=1 wtgdl=0.5 th train_vid_weighted.lua
# !!Diverges: DATA_ROOT=dataset/train nBottleneck=5000 name=testl2gdl loadSize=240 gpu=2 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=0.4 wtl2=1 wtgdl=0.5 th train_vid_weighted.lua

# test small loadSize=200
# DATA_ROOT=dataset/train nBottleneck=5000 name=testgt1_ldsz200 loadSize=200 gpu=2 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=0.8 th train_vid_weighted.lua

### 0907 ###
# test train_wholeim_input.lua, in which the network is very wide (not deep) and predLen=1 should not change
# RESULT: training is very struggling, and seems hard to converge to a good point, should test with other *learning rate*.
# DATA_ROOT=dataset/train name=test_bignet_crop loadSize=360 gpu=3 display_port=8002 display_iter=15 th train_wholeim_input.lua

### 0908 ###
# test initializer or a rnn intuition. In train_vid_weighted.lua set withInit=1
# DATA_ROOT=dataset/train nBottleneck=5000 name=test_init loadSize=280 gpu=3 nThreads=4 predLen=1 batchSize=64 display_port=8002 lr=0.0001 weight_nomask=1 loadName=test_init loadIter=40 withInit=1 initName=checkpoints/0906/testgt1_80_net_G.t7 th train_vid_weighted.lua

### 0910 ###
# DATA_ROOT=dataset/train nBottleneck=5000 name=testgt1_ldsz240 loadSize=240 gpu=3 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=1 loadName=0907/testgt1_ldsz200 loadIter=60 th train_vid_weighted.lua

### 0912 ###
# DATA_ROOT=dataset/train nBottleneck=5000 name=test_0912 loadSize=240 gpu=3 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=0.5 th train_vid_weighted.lua

### 0918 ### 4 channel input
ntrain=36000 DATA_ROOT=dataset/train nBottleneck=5000 name=test_0918 loadSize=180 gpu=3 nThreads=4 predLen=1 batchSize=64 display_port=8002 weight_nomask=1 th train_4channel.lua