import cv2
import mediapipe as mp
import time #delay
mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
#detector
hand=mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.7)
cam=cv2.VideoCapture(0)
photo_taken=False
last_capture=0
#funcion to check whether your hand is open
def is_open(hand_landmarks):
    tips=[8,12,16,20]
    fingers_up=0
    #checking tip position
    for tip in tips:
        if hand_landmarks.landmark[tip].y<hand_landmarks.landmark[tip-2].y:
            fingers_up+=1
    thump_up=hand_landmarks.landmark[4].x>hand_landmarks.landmark[3].x
    return fingers_up==4 and thump_up
#main program
while True:
    success,frame=cam.read()
    if not success:
        print("Error: Couldn't read webcam")
        break
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #detect the hands
    results=hand.process(rgb)
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            #draw the landmark
            mp_draw.draw_landmarks(frame,landmark,mp_hands.HAND_CONNECTIONS)
            #check if hand is open
            if is_open(landmark):
                cv2.putText(frame,"OPEN PALM DETECTED",(20,50),cv2.FONT_ITALIC,1,(255,255,255),2)
                #store the time since last photo taken
                current_time=time.time()
                #3 second delay between photos
                if current_time-last_capture>3:
                    #save the photo taken
                    filename=f"Photo_{int(current_time)}.jpg"
                    cv2.imwrite(filename,frame)
                    print(f"Photo Saved:{filename}")
                    last_capture=current_time
    #display the output
    cv2.imshow("Photo App",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()         
     