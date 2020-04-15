import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])
y_data = np.array([
    [1,0,0], #기타
    [0,1,0], #포유류
    [0,0,1], #조류
    [1,0,0],
    [1,0,0],
    [0,0,1]
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#가중치 변수 2 
W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))
#편향 변수 2개
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

#활성화 함수
L1 = tf.add(tf.matmul(X, W1),b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)
#model = tf.nn.softmax(L)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

#다양한 최적화 함수를 사용해볼것
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate =0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(100):
        sess.run(train_op, feed_dict={X:x_data, Y:y_data})

        if(step + 1) % 10 == 0:
            print(step +1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    prediction = tf.argmax(model, axis=1)
    target = tf.argmax(Y, axis=1)
    print('prediction:', sess.run(prediction,feed_dict={X:x_data}))
    print('target:', sess.run(target,feed_dict={Y:y_data}))

    is_correct = tf.equal(prediction,target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('accuracy: %2f'%sess.run(accuracy*100,feed_dict={X:x_data, Y:y_data}))

