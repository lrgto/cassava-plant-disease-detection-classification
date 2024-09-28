import os
import shutil
import pandas as pd

labels = pd.read_csv('Cyssava/utrain.csv')

# Classes are :
# 0: 'Cassava Bacterial Blight (CBB)',
# 1: 'Cassava Brown Streak Disease (CBSD)',
# 2: 'Cassava Green Mottle (CGM)',
# 3: 'Cassava Mosaic Disease (CMD)',
# 4: 'Healthy'

CBB = labels.loc[labels['labels'] == 0]
CBSD = labels.loc[labels['labels'] == 1]
CGM = labels.loc[labels['labels'] == 2]
CMD = labels.loc[labels['labels'] == 3]
HEALTHY = labels.loc[labels['labels'] == 4]

def moveTest(files, category):
  srcpath = 'Cyssava/utrain_images/'
  destpath = 'Cyssava/' + category + '/'
  if (not os.path.exists(destpath)):
    os.makedirs(destpath)
  for file in files:
    try:
      shutil.move(srcpath + file, destpath + file)
    except FileNotFoundError:
      if os.path.exists(destpath + file):
        # file already moved, can be ignored
        # print(file+' already moved')
        continue
      else:
        print('could not find ' + file)
        continue


moveTest(CBB.loc[:, "image_id"], "CBB")
moveTest(CBSD.loc[:, "image_id"], "CBSD")
moveTest(CGM.loc[:, "image_id"], "CGM")
moveTest(CMD.loc[:, "image_id"], "CMD")
moveTest(HEALTHY.loc[:, "image_id"], "HEALTHY")