"""
# indices are interpreted from right to left
# rightmost stored continuously, then stride
#
# VERTICAL-HORIZONTAL:
#       Tensorflow: shape [4,3] is a 2D image: row major format, 4 rows 3 columns
#       Matrix notation:  1st row index, 2nd column index
# HORIZONTAL-VECTICAL:
#       Monitor/jpeg: compared to monitor standrad: 640x480 first width(columns) then height(rows)
#       Geometry notation: plotting (x,y) first horizontal then vertical
#
# Exampe: tensor [batch,height,width,channel]
# So corresponding channel pixels are stored continuously
"""
import tensorflow as tf

# Initialize session.
session = tf.Session()
tf.global_variables_initializer()

temp3= tf.constant([0,1,2,3],tf.float32)
temp4= tf.reshape(temp3,[2,2])
print(session.run(temp4))
temp3= tf.constant([0,1,2,3,4,5,6,7],tf.float32)
temp4= tf.reshape(temp3,[2,2,2])
print(session.run(temp4))
print(session.run(tf.reshape(temp4,[4,2])))