import tensorflow as tf

print("TF version:", tf.__version__)
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
#physical_devices = session_config.list_physical_devices('GPU') 
#for device in physical_devices:
#session_config.experimental.set_memory_growth(device, True)
#session = tf.Session(config=session_config)
session = tf.Session(config=session_config)

with session as sess:
    # shape [512, 2] x [2, 768]
    a = tf.random.uniform([256, 2], dtype=tf.float32)
    b = tf.random.uniform([2, 256], dtype=tf.float32)
    c = tf.matmul(a, b)
    out = sess.run(c)
    print("Output shape:", out.shape)
    print("Success!")