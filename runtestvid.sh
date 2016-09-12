# This script uses test_vid_wholeim.lua to test and generate gif of whole frames.
# possible parameters: 
#  predLen = 40,
#  batchSize = 1,
#  inputLen = 1,
#  net = 'checkpoints/' .. '',
#  name = 'test1',        
#  gpu = 3,              
#  nc = 3,                
#  display = 1,           
#  loadSize = 360,        
#  fineSize = 128,        
#  nThreads = 1,          
#  maskName = 'maskppp.png',
#  maskValue = 110/255, 
#  manualSeed = 0,       
#  overlapPred = 0,    
#  DATA_ROOT = ''

#DATA_ROOT=dataset/target name=40im1 predLen=40 inputLen=1 maskName=maskppp.png loadSize=360 net=checkpoints/maskppp/maskppp_140_net_G.t7 th test_vid_wholeim.lua
#DATA_ROOT=dataset/train name=40im3 predLen=40 inputLen=1 maskName=mask6p.png loadSize=240 net=checkpoints/test_gdl_240_60_net_G.t7 th test_vid_wholeim.lua
#DATA_ROOT=dataset/target name=40im2 predLen=40 inputLen=1 maskName=maskppp.png loadSize=360 net=checkpoints/maskppp/maskppp_140_net_G.t7 th test_vid_wholeim.lua
#DATA_ROOT=dataset/target name=40im2 predLen=40 net=checkpoints/train1_120_net_G.t7 th test_vid_wholeim.lua
#DATA_ROOT=dataset/target name=40im3 predLen=40 net=checkpoints/train1_120_net_G.t7 th test_vid_wholeim.lua

# maskValue=0
#DATA_ROOT=dataset/target name=40im1 predLen=40 inputLen=1 maskName=maskppp.png loadSize=360 maskValue=0 net=checkpoints/changemaskvalue/changemaskvalue_40_net_G.t7 th test_vid_wholeim.lua

### 0906 ###
# testgt1
# DATA_ROOT=dataset/target name=40im4 loadSize=280 gpu=1 nThreads=4 predLen=40 inputLen=1 maskName=maskppp.png net=checkpoints/testgt1_80_net_G.t7 th test_vid_wholeim.lua

##### 0911 ### SERIAL TESTS ###
### Test arrangement: 3 tests for each model with same manualSeed, saved by test names and loadsizes. Use maskpppp.png as mask.
### net= 'checkpoints/' .. net
### Find good seed for test. try 10 tests. net=others/testgt1_60_net_G.t7 loadSize=350
# name=test001 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test002 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test003 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test004 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test005 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test006 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test007 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test008 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test009 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test010 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test011 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua
# name=test012 net=0906/testgt1_60_net_G.t7 loadSize=360 predLen=40 inputLen=1 th test_vid_wholeim.lua

### Single image tests arranged by loadsize
# 360
# name=results_laowang/naive_model/360/1 net=maskppp/maskppp_140_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/naive_model/360/2 net=maskppp/maskppp_140_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/naive_model/360/3 net=maskppp/maskppp_140_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/naive_model/360/4 net=maskppp/maskppp_140_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/naive_model/360/5 net=maskppp/maskppp_140_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=4791 th test_vid_wholeim.lua

# 240
# name=results_laowang/naive_model/240/1 net=0909/testgt1_ldsz240_120_net_G.t7 loadSize=240 predLen=40 inputLen=1 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/naive_model/240/2 net=0909/testgt1_ldsz240_120_net_G.t7 loadSize=240 predLen=40 inputLen=1 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/naive_model/240/3 net=0909/testgt1_ldsz240_120_net_G.t7 loadSize=240 predLen=40 inputLen=1 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/naive_model/240/4 net=0909/testgt1_ldsz240_120_net_G.t7 loadSize=240 predLen=40 inputLen=1 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/naive_model/240/5 net=0909/testgt1_ldsz240_120_net_G.t7 loadSize=240 predLen=40 inputLen=1 manualSeed=4791 th test_vid_wholeim.lua

# 200
# name=results_laowang/naive_model/200/1 net=0907/testgt1_ldsz200_60_net_G.t7 loadSize=200 predLen=40 inputLen=1 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/naive_model/200/2 net=0907/testgt1_ldsz200_60_net_G.t7 loadSize=200 predLen=40 inputLen=1 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/naive_model/200/3 net=0907/testgt1_ldsz200_60_net_G.t7 loadSize=200 predLen=40 inputLen=1 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/naive_model/200/4 net=0907/testgt1_ldsz200_60_net_G.t7 loadSize=200 predLen=40 inputLen=1 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/naive_model/200/5 net=0907/testgt1_ldsz200_60_net_G.t7 loadSize=200 predLen=40 inputLen=1 manualSeed=4791 th test_vid_wholeim.lua

# 480
# name=results_laowang/naive_model/480/1 net=0909/testgt1_ldsz480_120_net_G.t7 loadSize=480 predLen=40 inputLen=1 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/naive_model/480/2 net=0909/testgt1_ldsz480_120_net_G.t7 loadSize=480 predLen=40 inputLen=1 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/naive_model/480/3 net=0909/testgt1_ldsz480_120_net_G.t7 loadSize=480 predLen=40 inputLen=1 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/naive_model/480/4 net=0909/testgt1_ldsz480_120_net_G.t7 loadSize=480 predLen=40 inputLen=1 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/naive_model/480/5 net=0909/testgt1_ldsz480_120_net_G.t7 loadSize=480 predLen=40 inputLen=1 manualSeed=4791 th test_vid_wholeim.lua

### Multiple image tests arranged by loadsize
# 360
# name=results_laowang/multi_image/360/1 net=vid_fr4_wt1_ldsz350/vid_fr4_wt1_120_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/multi_image/360/2 net=vid_fr4_wt1_ldsz350/vid_fr4_wt1_120_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/multi_image/360/3 net=vid_fr4_wt1_ldsz350/vid_fr4_wt1_120_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/multi_image/360/4 net=vid_fr4_wt1_ldsz350/vid_fr4_wt1_120_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/multi_image/360/5 net=vid_fr4_wt1_ldsz350/vid_fr4_wt1_120_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=4791 th test_vid_wholeim.lua

# 480
# name=results_laowang/multi_image/480/1 net=vid_fr4_wt1_ldsz480/vid_fr4_wt1_ldsz480_100_net_G.t7  loadSize=480 predLen=40 inputLen=4 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/multi_image/480/2 net=vid_fr4_wt1_ldsz480/vid_fr4_wt1_ldsz480_100_net_G.t7  loadSize=480 predLen=40 inputLen=4 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/multi_image/480/3 net=vid_fr4_wt1_ldsz480/vid_fr4_wt1_ldsz480_100_net_G.t7  loadSize=480 predLen=40 inputLen=4 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/multi_image/480/4 net=vid_fr4_wt1_ldsz480/vid_fr4_wt1_ldsz480_100_net_G.t7  loadSize=480 predLen=40 inputLen=4 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/multi_image/480/5 net=vid_fr4_wt1_ldsz480/vid_fr4_wt1_ldsz480_100_net_G.t7  loadSize=480 predLen=40 inputLen=4 manualSeed=4791 th test_vid_wholeim.lua

# # 720
# name=results_laowang/multi_image/720/1 net=vid_fr4_wt1_ldsz720/vid_fr4_wt1_ldsz720_40_net_G.t7  loadSize=720 predLen=40 inputLen=4 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/multi_image/720/2 net=vid_fr4_wt1_ldsz720/vid_fr4_wt1_ldsz720_40_net_G.t7  loadSize=720 predLen=40 inputLen=4 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/multi_image/720/3 net=vid_fr4_wt1_ldsz720/vid_fr4_wt1_ldsz720_40_net_G.t7  loadSize=720 predLen=40 inputLen=4 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/multi_image/720/4 net=vid_fr4_wt1_ldsz720/vid_fr4_wt1_ldsz720_40_net_G.t7  loadSize=720 predLen=40 inputLen=4 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/multi_image/720/5 net=vid_fr4_wt1_ldsz720/vid_fr4_wt1_ldsz720_40_net_G.t7  loadSize=720 predLen=40 inputLen=4 manualSeed=4791 th test_vid_wholeim.lua

### GDL 
# name=results_laowang/gdl_loss/360/01 net=test_gdl/test_gdl_80_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/02 net=test_gdl/test_gdl_80_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/03 net=test_gdl/test_gdl_80_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/04 net=test_gdl/test_gdl_80_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/05 net=test_gdl/test_gdl_80_net_G.t7 loadSize=360 predLen=40 inputLen=1 manualSeed=4791 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/11 net=test_gdl/test_gdl_manyim_80_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=24751 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/12 net=test_gdl/test_gdl_manyim_80_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=5079 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/13 net=test_gdl/test_gdl_manyim_80_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=7814 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/14 net=test_gdl/test_gdl_manyim_80_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=37062 th test_vid_wholeim.lua
# name=results_laowang/gdl_loss/360/15 net=test_gdl/test_gdl_manyim_80_net_G.t7 loadSize=360 predLen=40 inputLen=4 manualSeed=4791 th test_vid_wholeim.lua


### More Context


### Initializer
name=results_laowang/with_init/280/1 initName=0906/testgt1_80_net_G.t7 net=0909/test_init_60_net_G.t7 loadSize=280 predLen=40 inputLen=1 manualSeed=24751 th test_vid_wholeim.lua
name=results_laowang/with_init/280/2 initName=0906/testgt1_80_net_G.t7 net=0909/test_init_60_net_G.t7 loadSize=280 predLen=40 inputLen=1 manualSeed=5079 th test_vid_wholeim.lua
name=results_laowang/with_init/280/3 initName=0906/testgt1_80_net_G.t7 net=0909/test_init_60_net_G.t7 loadSize=280 predLen=40 inputLen=1 manualSeed=7814 th test_vid_wholeim.lua
name=results_laowang/with_init/280/4 initName=0906/testgt1_80_net_G.t7 net=0909/test_init_60_net_G.t7 loadSize=280 predLen=40 inputLen=1 manualSeed=37062 th test_vid_wholeim.lua
name=results_laowang/with_init/280/5 initName=0906/testgt1_80_net_G.t7 net=0909/test_init_60_net_G.t7 loadSize=280 predLen=40 inputLen=1 manualSeed=4791 th test_vid_wholeim.lua