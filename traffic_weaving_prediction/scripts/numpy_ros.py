import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Float64MultiArray,\
                         Int32MultiArray, Int64MultiArray,\
                         MultiArrayLayout, MultiArrayDimension

def numpy_to_multiarray(A):
    strides = np.cumprod(A.shape[::-1])[::-1]    # unused, but conforms to the ROS spec
    layout = MultiArrayLayout(dim=[MultiArrayDimension(size=d, stride=s) for d, s in zip(A.shape, strides)])
    if A.dtype == np.float32:
        return Float32MultiArray(layout, A.reshape(-1).tolist())
    elif A.dtype == np.float64:
        return Float64MultiArray(layout, A.reshape(-1).tolist())
    elif A.dtype == np.int32:
        return Int32MultiArray(layout, A.reshape(-1).tolist())
    elif A.dtype == np.int64:
        return Int64MultiArray(layout, A.reshape(-1).tolist())
    else:
        raise TypeError

def multiarray_to_numpy(msg):
    if isinstance(msg, Float32MultiArray):
        dtype = np.float32
    elif isinstance(msg, Float64MultiArray):
        dtype = np.float64
    elif isinstance(msg, Int32MultiArray):
        dtype = np.int32
    elif isinstance(msg, Int64MultiArray):
        dtype = np.int64
    else:
        raise TypeError
    return np.array(msg.data).astype(dtype).reshape([d.size for d in msg.layout.dim])