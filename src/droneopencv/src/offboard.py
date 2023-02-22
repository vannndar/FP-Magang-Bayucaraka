#! /usr/bin/env python3
import rospy
import numpy as np
import math
import time
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

import face_recognition
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
import cv2

#roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"

current_state = State()

def check_move_two(lmList1, lmList2): #check controller condition
    if (abs(((lmList1[4][1] + lmList1[8][1]) /2) - centerRightX) <= 25):
        fSpeed = 0
    else : 
        fSpeed = (centerRightX - ((lmList1[4][1] + lmList1[8][1]) /2))/10

    if (abs(((lmList1[4][0] + lmList1[8][0]) /2) - centerRightY) <= 25):
        sSpeed = 0
    else : 
        sSpeed = (((lmList1[4][0] + lmList1[8][0]) /2) - centerRightY)/10
    if (abs(((lmList2[4][1] + lmList2[8][1]) /2) - centerLeftX) <= 25):
        uSpeed = 0
    else : 
        uSpeed = (centerLeftX - ((lmList2[4][1] + lmList2[8][1]) /2) )/40
    if (abs(((lmList2[4][0] + lmList2[8][0]) /2) - centerLeftY) <= 25):
        rSpeed = 0
    else : 
        rSpeed = (((lmList2[4][0] + lmList2[8][0]) /2) - centerLeftY)/60
    do_publish(fSpeed,sSpeed,uSpeed,rSpeed) #publish velocity body frame


def state_cb(msg): #take state massage
    global current_state
    current_state = msg

def callbackpose(msg): #take call massage
    global pose_now
    pose_now = msg

def callbackraw(msg): #take call massage
    global current_raw
    current_raw= msg

def check_done_takeoff(pos_z) -> bool: #checking if drone has reached setpoint
    z = False
    if(abs(pose_now.pose.position.z - pos_z) <= 0.1):
        z = True
    #return true if all axis has reached setpoint
    if (z): return True
    else: return False

def do_publish(vel_x, vel_y, vel_z, vel_yaw): #set velocity and publish
    raw.coordinate_frame = PositionTarget.FRAME_BODY_NED
    raw.type_mask = 0b0000011111000111
    raw.velocity.x = vel_x
    raw.velocity.y = vel_y
    raw.velocity.z = vel_z
    raw.yaw_rate   = vel_yaw
    local_raw_pub.publish(raw)

def set_mode():
    #set mode
    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    #arming
    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    #record time before while
    last_req = rospy.Time.now()

    while(not rospy.is_shutdown()):
        #checking if drone mode
        if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if(set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")
            last_req = rospy.Time.now()
        #check drone arm state
        else:
            if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if(arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")
                last_req = rospy.Time.now()
        #publish setpoint
        local_pos_pub.publish(pose)
        #checking if drone has reached setpoint
        if(check_done_takeoff(pose.pose.position.z)):
            break
        rate.sleep()

def recognized():#recognized user
    # Load sample picture and learn how to recognize it.
    ivan_image = face_recognition.load_image_file("ivan.jpg")
    ivan_face_encoding = face_recognition.face_encodings(ivan_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        ivan_face_encoding
    ]
    known_face_names = [
        "ivan"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    face = False

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
                if(name == 'ivan'):
                    print("face recognized"+", Hello", name)
                    print("Start flying")
                    face = True
                    break

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if (cv2.waitKey(1) & 0xFF == ord('q')) or face:
            cv2.destroyAllWindows()
            break

def narto():
    #back to home
    while True:
        local_pos_pub.publish(pose)
        if(check_done_takeoff(pose.pose.position.z)):
            break
    pi = math.pi
    #move half circle speed 1
    while True:
        do_publish(1,0,0,-0.5)
        if((current_raw.yaw >= (pi-0.12) or current_raw.yaw <= -(pi-0.12))):
            do_publish(0,0,0,0)
            break
    #move half circle speed 2
    while True:
        do_publish(2,0,0,-0.5)
        if((-(0.12)<= current_raw.yaw <= (0.12))):
            do_publish(0,0,0,0)
            break
    #move half circle speed 4
    while True:
        do_publish(4,0,0,-0.5)
        if((current_raw.yaw >= (pi-0.12) or current_raw.yaw <= -(pi-0.12))):
            do_publish(0,0,0,0)
            break
    #move half circle speed 6
    while True:
        do_publish(6,0,0,-0.5)
        if(((pi/4-0.12)<= current_raw.yaw <= (pi/4+0.12))):
            do_publish(0,0,0,0)
            break
    while True:
        do_publish(0,0,0,0.5)
        if(((3*pi/4-0.12)<= current_raw.yaw <= (3*pi/4+0.12))):
            do_publish(0,0,0,0)
            break
    #move forward
    now = pose_now.pose.position.x
    while True:
        do_publish(1,0,0,0)
        if(pose_now.pose.position.x <= now-2):
            do_publish(0,0,0,0)
            break
    
def qrcode():
    #checking massage
    for s in list(decoded_info):
        if s == "narto":
            #destroy windows and realease video
            cv2.destroyAllWindows()
            cap.release()
            narto()

if __name__ == "__main__":
    rospy.init_node("offb_node_py")
    state_sub = rospy.Subscriber("mavros/state", State, callback = state_cb)
    pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, callbackpose)
    raw_sub = rospy.Subscriber("/mavros/setpoint_raw/target_local", PositionTarget, callbackraw)
    local_vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    local_raw_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
    
    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)    

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while(not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()

    #assign pose for PoseStamped(), pose_now for PoseStamped() abd velo for Twist() 
    pose = PoseStamped()
    pose_now = PoseStamped()
    current_raw = PositionTarget()
    velo = Twist()
    raw = PositionTarget()


    qcd = cv2.QRCodeDetector()
    cap = cv2.VideoCapture(0)
    success, img = cap.read()

    recognized()

    #set take off height
    pose.pose.position.z = 2

    # Send a few setpoints before starting
    for i in range(100):   
        if(rospy.is_shutdown()):
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    set_mode()

    #determine handdetector and facedetector
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    detectorface = FaceDetector()

    #setting some variable
    right = False
    left = False
    centerRightX = 0
    centerRightY = 0
    centerLeftX = 0
    centerLeftY = 0


    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, imgh = detector.findHands(img)  # with draw
        success_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(img)
        if (success_qr):
            qrcode()
            cap = cv2.VideoCapture(0)

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            handType1 = hand1["type"]  # Handtype Left or Right
            secHand = False

            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"
                secHand = True
            
            if (handType1 == "Right"): #active hand == right
                #check center and save
                if ( abs(lmList1[4][1]-lmList1[8][1]) <= 5 and right == False):
                    print(handType1 + " Recognized")
                    centerRightX = int((lmList1[4][1] + lmList1[8][1]) /2)
                    centerRightY = int((lmList1[4][0] + lmList1[8][0]) /2)
                    right = True
                    
                #check center and save
                if(secHand):
                    if(abs(lmList2[4][1]-lmList2[8][1] <= 5 and left == False)):
                        print(handType2 + " Recognized")
                        centerLeftX = int((lmList2[4][1]+lmList2[8][1])/2)
                        centerLeftY = int((lmList2[4][0]+lmList2[8][0])/2)
                        left = True

                #print border for 0 velocity
                if(right and left):
                    cv2.rectangle(img, pt1=(centerRightY+25,centerRightX+25), pt2=(centerRightY-25,centerRightX-25), color=(255,0,0), thickness=2)
                    cv2.rectangle(img, pt1=(centerLeftY+25,centerLeftX+25), pt2=(centerLeftY-25,centerLeftX-25), color=(255,0,0), thickness=2)

                #check if hands are ready to control drone
                if(secHand):
                    if(right and 
                       left and 
                       abs(lmList1[4][1]-lmList1[8][1]) <= 10 and 
                       abs(lmList2[4][0]-lmList2[8][0]) <= 10 and
                       abs(lmList2[4][1]-lmList2[8][1]) <= 10 and
                       abs(lmList2[4][0]-lmList2[8][0]) <= 10 ):
                        check_move_two(lmList1,lmList2)

            else: #active hand == left
                #check center and save
                if ( abs(lmList1[4][1]-lmList1[8][1]) <= 5 and left == False):
                    print(handType1 + "Recognized")
                    centerLeftX = int((lmList1[4][1] + lmList1[8][1]) /2)
                    centerLeftY = int((lmList1[4][0] + lmList1[8][0]) /2)
                    left = True
                    
                #check center and save
                if(secHand):
                    if(abs(lmList2[4][1]-lmList2[8][1] <= 5 and right == False)):
                        print(handType2 + "Recognized")
                        centerRightX = int((lmList2[4][1]+lmList2[8][1])/2)
                        centerRightY  = int((lmList2[4][0]+lmList2[8][0])/2)
                        right = True

                #print border for 0 velocity
                if(right and left):
                    cv2.rectangle(img, pt1=(centerRightY+25,centerRightX+25), pt2=(centerRightY-25,centerRightX-25), color=(255,0,0), thickness=2)
                    cv2.rectangle(img, pt1=(centerLeftY+25,centerLeftX+25), pt2=(centerLeftY-25,centerLeftX-25), color=(255,0,0), thickness=2)

                #check if hands are ready to control drone
                if(secHand):
                    if(right and 
                       left and 
                       abs(lmList1[4][1]-lmList1[8][1]) <= 10 and 
                       abs(lmList2[4][0]-lmList2[8][0]) <= 10 and
                       abs(lmList2[4][1]-lmList2[8][1]) <= 10 and
                       abs(lmList2[4][0]-lmList2[8][0]) <= 10 ):
                        check_move_two(lmList2,lmList1)
        #show image
        cv2.imshow("Image", img)
        #show wait for end program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break