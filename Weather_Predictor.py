import tensorflow as tf
from Data_Retriever import build_data_subset


# Function to measure accuracy by comparing actual model output to expected correct answer labels
def measure_accuracy(actual, expected):
    num_correct = 0
    for i in range(len(actual)):
        actual_value = actual[i]
        expected_value = expected[i]
        if actual_value[0] >= actual_value[1] and expected_value[0] >= expected_value[1]:
            num_correct += 1
        elif actual_value[0] <= actual_value[1] and expected_value[0] <= expected_value[1]:
            num_correct += 1
    return (num_correct / len(actual)) * 100
# Number of factors to apply to our prediction
input_shape = 4

x_train, y_train = build_data_subset('2018_weather_data.csv', 1, 37)
x_test, y_test = build_data_subset('2018_weather_data.csv', 38, 7)

# y = Wx + b
# Input node to feed in any number of data points for training/testing
x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_shape], name='x_input')
# Input node to feed in any number of correct labels for training purposes
y_input = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y_input')

# Variable node to represent weights, initial value of a bunch of ones
W = tf.Variable(initial_value=tf.ones(shape=[input_shape, 2]), name='W')
# Variable node to represent biases, initial value of a bunch of ones
b = tf.Variable(initial_value=tf.ones(shape=[2]), name='b')

# Output node to perform the calculation and fit a line through input factors and output labels
# Call upon this to get model output
y_output = tf.add(tf.matmul(x_input, W), b, name='y_output')

# Loss function to measure difference between expected correct and actual model output answers
loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_output)))
# Adam optimizer will attempt to minimize loss by adjusting variable values at learning rate of 0.005
optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)

saver = tf.train.Saver()
# Create the tensorflow session and initialize global variables
session = tf.Session()
session.run(tf.global_variables_initializer())

tf.train.write_graph(session.graph_def, '.', 'weather_prediction.pbtxt', False)

# Run the training data 20000 times (epochs) by teaching model what correct answers are given certain inputs
for _ in range(20000):
    session.run(optimizer, feed_dict={x_input: x_train, y_input: y_train})

saver.save(session, 'weather_prediction.ckpt')

# Print out the accuracy of test and train data
print(measure_accuracy(session.run(y_output, feed_dict={x_input: x_train}), y_train))
print(measure_accuracy(session.run(y_output, feed_dict={x_input: x_test}), y_test))
