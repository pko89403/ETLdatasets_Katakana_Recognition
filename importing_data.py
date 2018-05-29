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
    train_label.append(int(data[1]))

for valid in validList:
    data = valid.strip().split()
    valid_img.append(data[0])
    valid_label.append(int(data[1]))

for test in testList:
    data = test.strip().split()
    test_img.append(data[0])
    test_label.append(int(data[1]))

train_imgs = tf.constant(train_img)
train_labels = tf.constant(train_label)

valid_imgs = tf.constant(valid_img)
valid_labels = tf.constant(valid_label)

test_imgs = tf.constant(test_img)
test_labels = tf.constant(test_label)

train_data = Dataset.from_tensor_slices((train_imgs, train_labels))
valid_data = Dataset.from_tensor_slices((valid_imgs, valid_labels))
test_data = Dataset.from_tensor_slices((test_imgs, test_labels))

NUM_CLASSES=48

def input_parser(train_img, train_label):
    one_hot = tf.one_hot(train_label, NUM_CLASSES)
    img_file = tf.read_file(train_img)
    img_decoded = tf.image.decode_image(img_file, channels=1)
    img_resize = tf.image.resize_image_with_crop_or_pad(img_decoded,target_width=64,target_height=64)
    return img_resize, one_hot

train_data.map(input_parser)
valid_data.map(input_parser)
test_data.map(input_parser)

# create Tensorflow Iterator object
iterator = Iterator.from_structure(train_data.output_types,
                                   train_data.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(train_data)

"""
with tf.Session() as sess:
    sess.run(training_init_op)

    while True:
        try:

            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
"""