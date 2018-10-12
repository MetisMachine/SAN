#!/usr/bin/python3
import sys
import os
import random
import glob
import string

try:
  from lxml import etree as ET
except ImportError:
  print('install lxml using pip')
  print('pip install lxml')

# create XML from annotations
def createXml(imageNames, xmlName, numPoints, verbose=True):
  # create a root node names dataset
  dataset = ET.Element('dataset')
  # create a child node "name" within root node "dataset"
  ET.SubElement(dataset, "name").text = "Training Faces"
  # create another child node "images" within root node "dataset"
  images = ET.SubElement(dataset, "images")

  # print information about xml filename and total files
  numFiles = len(imageNames)
  print('{0} : {1} files'.format(xmlName, numFiles))

  # iterate over all files
  for k, imageName in enumerate(imageNames):
    # print progress about files being read
    if verbose: print('{}:{} - {}'.format(k+1, numFiles, imageName))

    rect_name = imageName + '_rect.txt'
    points_name = imageName + '_bv' + numPoints + 'c.txt'

    if os.path.exists(rect_name) and os.path.exists(points_name):
      # read rectangle file corresponding to image
      with open(rect_name, 'r') as file:
        rect = file.readline()
      rect = rect.split()
      left, top, width, height = rect[0:4]

      # create a child node "image" within node "images"
      # this node will have annotation data for an image
      image = ET.SubElement(images, "image", file=os.path.abspath(imageName+'.jpg'))
      # create a child node "box" within node "image"
      # this node has values for bounding box or rectangle of face
      box = ET.SubElement(image, 'box', top=top, left=left, width=width, height=height)

      # read points file corresponding to image
      with open(points_name, 'r') as file:
        for i, point in enumerate(file):
          x, y = point.split()
          # points annotation file has coordinates in float
          # but we want them to be in int format
          x = str(int(float(x)))
          y = str(int(float(y)))
          # name is the facial landmark or point number, starting from 0
          name = str(i).zfill(2)
          # create a child node "parts" within node "box"
          # this node has values for facial landmarks
          ET.SubElement(box, 'part', name=name, x=x, y=y)

  # finally create an XML tree
  tree = ET.ElementTree(dataset)

  print('writing on disk: {}'.format(xmlName))
  # write XML file to disk. pretty_print=True indents the XML to enhance readability
  tree.write(xmlName, pretty_print=True, xml_declaration=True, encoding="UTF-8")


def createSANInputFiles(directoryPath, imageNames, fileName, numPoints, verbose=True):
  
  numFiles = len(imageNames)
  
  print(directoryPath)
  print(fileName)
  
  for k, imageName in enumerate(imageNames):
    # print progress about files being read
    if verbose: print('{}:{} - {}'.format(k+1, numFiles, imageName))

    rect_name = os.path.join(directoryPath, imageName + '_rect.txt')
    points_name = os.path.join(directoryPath, imageName + '_bv' + numPoints + 'c.txt')
    picture_name = os.path.join(directoryPath, imageName + '.jpg')

    if os.path.exists(rect_name) and os.path.exists(points_name):
      # read rectangle file corresponding to image
      with open(rect_name, 'r') as file:
        rect = file.readline()
      rect = rect.split()
      left, top, width, height = rect[0:4]
      
      # Add and subtract points to get min and max for SAN
      # Test this on SAN image? Not sure if this is right coordinate structure
      x1 = left
      y1 = top 
      x2 = str(int(left) + int(width))
      y2 = str(int(height) + int(top))
      
      box_str = " ".join([x1, y1, x2, y2])
      
      # Add the following to the top of each pts file to be compatible with SAN:
      # version: 1
      # n_points:  78
      # {
      # Add another } at the end
      with open(points_name, 'r+') as ptsFile: 
        content = ptsFile.read()
        # Some files were mysteriously going through this twice. 
        #if content.find('version') == -1:
        addons = "version: 1" + "\n" + "n_points:  " + numPoints + "\n" + "{" + "\n"
        ptsFile.seek(0, 0)
        ptsFile.write(addons + content + '}')
        #else:
          #print(points_name)
      
      
      fullImagePath = os.path.join(directoryPath, picture_name)
      fullPointsPath = os.path.join(directoryPath, points_name)
      
      with open(fileName, "a") as output: 
        output.write('{} {} {}\n'.format(fullImagePath, fullPointsPath, box_str))
    
    
if __name__ == '__main__':

  # read value to facial_landmark_data directory
  # and number of facial landmarks
  fldDatadir = sys.argv[1]
  numPoints = sys.argv[2]

  numMaleTrain = 550
  numFemaleTrain = 1300

  maleDir = os.path.join(fldDatadir, 'male')
  femaleDir = os.path.join(fldDatadir, 'female')
  
  maleRectPaths = glob.glob(os.path.join(maleDir, '*_rect.txt'))
  femaleRectPaths = glob.glob(os.path.join(femaleDir, '*_rect.txt'))

  maleImageNames = [os.path.splitext(x)[0].replace('_rect', '') for x in maleRectPaths if 'mirror' not in x]
  femaleImageNames = [os.path.splitext(x)[0].replace('_rect', '') for x in femaleRectPaths if 'mirror' not in x]

  random.seed(55)
  with open(os.path.join(fldDatadir, 'test.v1.txt')) as f:
    testFiles_v1 = [x.strip() for x in f.readlines() if 'mirror' not in x]

  maleTestFiles_v1 = [x for x in testFiles_v1 if x.startswith('male')]
  femaleTestFiles_v1 = [x for x in testFiles_v1 if x.startswith('female')]

  with open(os.path.join(fldDatadir, 'train.v1.txt')) as f:
    trainFiles_v1 = [x.strip() for x in f.readlines() if 'mirror' not in x]

  maleTrainFiles_v1 = [x for x in trainFiles_v1 if x.startswith('male')]
  femaleTrainFiles_v1 = [x for x in trainFiles_v1 if x.startswith('female')]

  # Earlier we had a dataset which we split into train and test.
  # Now that we have added more data, we will keep previous train/test data
  # and add files from new data to train/test.
  # To get which files are new we will substract all data from old data
  maleNewFiles = list(set(maleImageNames) - set(maleTrainFiles_v1) - set(maleTestFiles_v1))
  femaleNewFiles = list(set(femaleImageNames) - set(femaleTrainFiles_v1) - set(femaleTestFiles_v1))
  
  # Remove absolutle path here--not sure why it is added. 
  maleNewFiles2 = []
  femaleNewFiles2 = []
  
  for x in maleNewFiles:
    if x == os.path.abspath(x):
      y = x.split('/')
      z = y[-2:]
      maleNewFiles2.append("/".join(z))
    else:
      maleNewFiles2.append(x)
      
  for x in femaleNewFiles:
    if x == os.path.abspath(x):
      y = x.split('/')
      z = y[-2:]
      femaleNewFiles2.append("/".join(z))
    else:
      femaleNewFiles2.append(x)
      
    
  # Since we don't yet have enough train files even after adding more 
  # data. We are not going to add any more test files 
  # from newer set of female images.
  maleTestFiles = maleTestFiles_v1
  femaleTestFiles = femaleTestFiles_v1

  # Add training images from newer set of images
  maleTrainFiles_v2 = random.sample(maleNewFiles2, numMaleTrain - len(maleTrainFiles_v1))
  femaleTrainFiles_v2 = random.sample(femaleNewFiles2, numFemaleTrain - len(femaleTrainFiles_v1))

  maleTrainFiles = maleTrainFiles_v1 + maleTrainFiles_v2
  femaleTrainFiles = femaleTrainFiles_v1 + femaleTrainFiles_v2
  

  # extra check. not needed in this version.
  # make test set uniform with almost equal number of images from both male and female
  if len(maleTestFiles) > len(femaleTestFiles):
    maleTestFiles = random.sample(maleTestFiles, len(femaleTestFiles))
  else:
    femaleTestFiles = random.sample(femaleTestFiles, len(maleTestFiles))

  testFiles = maleTestFiles + femaleTestFiles
  trainFiles = maleTrainFiles + femaleTrainFiles

  print('male:: test v1:{}, train v1:{}, train v2:{}'.format(len(maleTestFiles_v1), len(maleTrainFiles_v1), len(maleTrainFiles_v2)))
  print('num:: male :: test - {} train - {}'.format(len(maleTestFiles), len(maleTrainFiles)))
  print('female:: test v1:{}, train v1:{}, train v2:{}'.format(len(femaleTestFiles_v1), len(femaleTrainFiles_v1), len(femaleTrainFiles_v2)))
  print('num:: female :: test - {} train - {}'.format(len(femaleTestFiles), len(femaleTrainFiles)))
  print('num:: test - {} train - {}'.format(len(testFiles), len(trainFiles)))

  # add mirrored image corresponding to each image in train and test list
  testFiles += [x + '_mirror' for x in testFiles]
  trainFiles += [x + '_mirror' for x in trainFiles]

  with open(os.path.join(fldDatadir, 'train.v2.txt'), 'w') as f:
    for p in trainFiles:
      f.write(p + '\n')
  with open(os.path.join(fldDatadir, 'test.v2.txt'), 'w') as g:
    for p in testFiles:
      g.write(p + '\n')

  #Create SAN input files
  createSANInputFiles(fldDatadir, trainFiles, os.path.join(fldDatadir, 'aedit-training.txt'), numPoints, verbose=False)
  createSANInputFiles(fldDatadir, testFiles, os.path.join(fldDatadir, 'aedit-testing.txt'), numPoints, verbose=False)

  # generate XML files for train and test data
  #createXml(trainFiles, os.path.join(fldDatadir, 'training_with_face_landmarks.xml'), numPoints, verbose=False)
  #createXml(testFiles, os.path.join(fldDatadir, 'testing_with_face_landmarks.xml'), numPoints, verbose=False)
  
  
