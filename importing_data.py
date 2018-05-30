import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

trainPATH='E:/python_workspace/ETLdatasets/train_dataList.csv'
validPATH='E:/python_workspace/ETLdatasets/valid_dataList.csv'
testPATH='E:/python_workspace/ETLdatasets/test_dataList.csv'

train_img = []
train_label = []
valid_img = []
valid_label = []
test_img = []
test_label = []

trainList = open(trainPATH, 'r')
validList = open(validPATH, 'r')
testList = open(testPATH, 'r')

for train in trainList:
    data = train.strip().split()
    train_img.append(data[0])
    if(int(data[1]) == 166): data[1]=174
    if(int(data[1]) == 168): data[1]=175
    if(int(data[1]) == 170): data[1]=176
    train_label.append(int(data[1])-174)


for valid in validList:
    data = valid.strip().split()
    valid_img.append(data[0])
    valid_label.append(int(data[1]))

for test in testList:
    data = test.strip().split()
    test_img.append(data[0])
    test_label.append(int(data[1]))

NUM_CLASSES=48

def input_parser(filename, filelabel):
    one_hot_res = tf.one_hot(filelabel, 48)
    print(one_hot_res)
    #img_file = tf.read_file(filename)
    #img_decoded = tf.image.decode_image(img_file,channels=1)
    #img_resize = tf.image.resize_image_with_crop_or_pad(img_decoded,target_width=64,target_height=64)
    return filename, one_hot_res


train_imgs = tf.constant(train_img)
train_labels = tf.constant(train_label)

valid_imgs = tf.constant(valid_img)
valid_labels = tf.constant(valid_label)

test_imgs = tf.constant(test_img)
test_labels = tf.constant(test_label)

train_data = Dataset.from_tensor_slices((train_imgs, train_labels))
valid_data = Dataset.from_tensor_slices((valid_imgs, valid_labels))
test_data = Dataset.from_tensor_slices((test_imgs, test_labels))

train_data = train_data.map(input_parser)
valid_data.map(input_parser)
test_data.map(input_parser)

# create Tensorflow Iterator object
iterator = train_data.make_initializable_iterator()
next_element = iterator.get_next()


sess = tf.Session()
sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
        break
