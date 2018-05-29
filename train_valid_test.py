import os
import numpy as np

imagesPATHS = 'E:/python_workspace/ETLdatasets/katakana/'

trainDataName = './train_dataLIst.csv'
validDataName = './valid_dataList.csv'
testDataName = './test_dataList.csv'

trainCount = 800
validCount = 300
testCount = 300


trainListFile = open(trainDataName,'w')
validListFile = open(validDataName,'w')
testListFile = open(testDataName,'w')

imageFolders=os.listdir(imagesPATHS)

for folder in imageFolders:
    imagesPATH = imagesPATHS + folder + '/'
    images = os.listdir(imagesPATH)
    np.random.shuffle(images)

    for tr in range(0,trainCount):
        name = imagesPATH + images[tr] + '\t' + folder + '\n'
        trainListFile.write(name)
    for vd in range(trainCount,trainCount+validCount):
        name = imagesPATH + images[vd] + '\t' + folder + '\n'
        validListFile.write(name)
    for te in range(trainCount+validCount,trainCount+validCount+testCount):
        name = imagesPATH + images[te] + '\t' + folder + '\n'
        testListFile.write(name)

trainListFile.close()
validListFile.close()
testListFile.close()