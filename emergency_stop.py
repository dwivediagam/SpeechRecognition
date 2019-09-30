from __future__ import print_function
from naoqi import ALProxy

ip = "172.16.21.208"
port = 9559

motion = ALProxy("ALMotion",ip,port)
motion.stopMove()












