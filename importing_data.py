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
    img_file = tf.read_file(filename)
    img_decoded = tf.image.decode_image(img_file,channels=1)
    img_resize = tf.image.resize_image_with_crop_or_pad(img_decoded,target_width=64,target_height=64)
    return img_resize, one_hot_res


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
train_data.shuffle(500).repeat().batch(30)

valid_data.map(input_parser)
test_data.map(input_parser)

# create Tensorflow Iterator object
iterator = Iterator.from_structure(train_data.output_types,
                                   train_data.output_shapes)

next_element = iterator.get_next()


train_init_op = iterator.make_initializer(train_data)


# model
def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    X = tf.cast(X, dtype='float')
    X = tf.reshape(X, shape=[-1,64,64,1])

    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1,1,1,1], padding='SAME'))         # shape = (?, 64, 64, 128)
    l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # shape = (?, 32, 32, 128)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME'))       # shape = (?, 32, 32, 256)
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # shape = (?, 16, 16, 256)
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME'))       # shape = (?, 16, 16, 512)
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')     # shape = (?,  8,  8, 512)
    l3 = tf.reshape(l3, shape=[-1, w4.get_shape().as_list()[0]])                     # reshape to (?, 32768)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

w = init_weights(shape=[3,3,1,128])     # 3*3*1 conv, 128 output
w2 = init_weights(shape=[3,3,128,256])  # 3*3*128 conv, 256 output
w3 = init_weights(shape=[3,3,256,512])  # 3*3*256 conv, 512 output
w4 = init_weights(shape=[512 * 8 * 8, 1000])
w_o = init_weights(shape=[1000, 48])

p_keep_conv = 0.8
p_keep_hidden = 0.5
logits = model(next_element[0],w,w2,w3,w4,w_o,p_keep_conv, p_keep_hidden)
# add the optimizer and loss
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=next_element[1], logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)
# get accuracy
prediction = tf.argmax(logits, 1)
actual = tf.argmax(next_element[1], 1)
equality = tf.equal(prediction, actual)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
init_op = tf.global_variables_initializer()

epochs   =  600
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(train_init_op)
    for i in range(epochs):
        l, _, acc = sess.run([loss, optimizer, accuracy])
        if i % 50 == 0:
            print("Epoch: {}, loss : {:.3f}, training_accuracy: {:.2f}%".format(i, l, acc * 100))