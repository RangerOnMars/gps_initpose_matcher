from ast import While
from cmath import cos, sin
from numpy import minimum
import utm
import rospy
import tf2_ros
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock
from tqdm import tqdm
from sensor_msgs.msg import NavSatFix


def generate_params_dict(key, params):
    for param in rospy.get_param_names():
        if param.split("/").count(key):
            param_key = param.split("/")[-1]
            params[param_key] = rospy.get_param(param)
    return params

def callback(gps, args):
    secs = gps.header.stamp
    tf_msg = args["tf"].lookup_transform(args["map_frame"],
                                      args["base_frame"],
                                      secs,
                                      rospy.Duration(args["tf_timeout"]))
    lat = gps.latitude
    lon = gps.longitude
    map_xy = [tf_msg.transform.translation.x,
              tf_msg.transform.translation.y]
    easting, northing, z_num, z_letter = utm.from_latlon(lat,lon)
    
    args["lock"].acquire()
    args["gpsxy_list"].append([northing, easting])
    args["mapxy_list"].append(map_xy)
    args["updated"] = True
    args["lock"].release()

def latlon2utm(lat,lon):
    easting, northing, z_num, z_letter = utm.from_latlon(lat,lon)
    return easting, northing, z_num, z_letter

def cost(x_gt, y_gt, x, y, tx, ty, theta, a):
    x_loss = x_gt - a * (cos(theta) - sin(theta)) * x - tx
    y_loss = y_gt - a * (sin(theta) + cos(theta)) * y - ty
    return (x_loss**2 + y_loss**2)**0.5

def main():
    params = {}
    params = generate_params_dict('map_utm_matcher', params)
    rospy.init_node('map_utm_matcher', anonymous=True)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
    callback_args = {}
    callback_args["tf"] = tfBuffer
    callback_args["gpsxy_list"] = []
    callback_args["mapxy_list"] = []
    callback_args["lock"] = Lock()
    callback_args["updated"] = False
    callback_args["tf_timeout"] = params["tf_timeout"]
    callback_args["max_len"] = params["max_data_len"]
    callback_args["map_frame"] = params["map_frame"]
    callback_args["base_frame"] = params["base_frame"]
    subscriber = rospy.Subscriber(params["gps_topic"], NavSatFix, callback, callback_args)
    
    #Display progress bar
    rospy.loginfo("Collecting tf and gps data...")
    for i in tqdm(range(params["max_data_len"])):
        while True:
            callback_args["lock"].acquire()
            if (callback_args["updated"]):
                callback_args["updated"] = False
                callback_args["lock"].release()
                time.sleep(1e-3)
                break
            else:
                callback_args["lock"].release()
                time.sleep(1e-3)
    subscriber.unregister()
    gpsxy_list = callback_args["gpsxy_list"]
    mapxy_list = callback_args["mapxy_list"]
    rospy.loginfo("Estimating Affine Matrix Using OpenCV")
    gps_xy = np.array(gpsxy_list)
    map_xy = np.array(mapxy_list)
    # print(gpsxy.shape)
    affine_mat, inliers = cv2.estimateAffinePartial2D(gps_xy,
                                                      map_xy,
                                                      method=cv2.RANSAC,
                                                      ransacReprojThreshold=15)
    affine_homomat = np.vstack((affine_mat,np.array([0,0,1])))
    rospy.loginfo("Evaluating result")
    errs = []
    pred_xy = []
    for i in tqdm (range(gps_xy.shape[0])):
        xy_from = gps_xy[i].reshape(-1,1)
        xy_to = map_xy[i].reshape(-1,1)
        pred = np.matmul(affine_homomat, np.vstack((xy_from,[1])))
        pred = pred[0:2]
        err = np.linalg.norm((xy_to - pred))
        errs.append(err)
        pred_xy.append(pred)
        print(xy_from)
        print(xy_to)
        print(pred)
        print(inliers[i])
        print(err)
        print("="*10)
    pred_xy = np.array(pred_xy)
    # plt.scatter(x=gps_xy[:,0], y=gps_xy[:,1],c="b")
    plt.title("Affine Result")
    plt.scatter(x=map_xy[:,0], y=map_xy[:,1],c="b")
    plt.scatter(x=pred_xy[:,0], y=pred_xy[:,1],c="r")
    plt.show()
    
    mse = np.array(errs).mean()
    rospy.loginfo("Done! Showing Result")
    rospy.loginfo("Affine matrix :\n {}".format(affine_homomat))
    rospy.loginfo("Mean Error :{} m".format(mse))
        

if __name__ == "__main__":
    main()