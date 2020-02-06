#Shuffle_batch
#min_after_dequeue defines how big a buffer we will radomly sample
#from --  bigger means better shuffling but slower start up and more
# memory used.
#capacity must be larger than min_after_dequeue and the amount larger
#determines the maximum we will prefetch. Recommendation :
#min_after_dequeue + (num_threads + a small safety margin) * batch_size

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch =
tf.train.shuffle_batch([example, label], batch_size = batch_size,
capacity = capacity, min_after_dequeue = min_after_dequeue)
