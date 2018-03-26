import tensorflow as tf
y = tf.placeholder("int64", [None], "y")
one_hot_y=tf.one_hot(y,10)
ce = tf.nn.softmax_cross_entropy_with_logits(one_hot_y, one_hot_y)
sess = tf.Session()
sess.run(ce, {y: []})
