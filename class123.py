import cv2
import mediapipe as mp

from  pynput.mouse import Button,Controller

import pyautogui
import math

mymouse=Controller()
pinch=False

state=None


video=cv2.VideoCapture(0)
width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width,height)

# to draw hands
myhands=mp.solutions.hands

# to display landmarks
mydrawing=mp.solutions.drawing_utils


hand_obj=myhands.Hands(min_detection_confidence=0.75,
                        min_tracking_confidence=0.75)
print("what is hand obj ",hand_obj)

# access the system width and height 
(screen_width,screen_height)=pyautogui.size()
print(screen_width,screen_height)
      

def countFingers(lst,myimage):
   count=0
   global pinch

   thresh=(lst.landmark[0].y*100-lst.landmark[9].y*100)/2
   #print("what is thresh ",thresh)#


   if(lst.landmark[5].y*100-lst.landmark[8].y*100)>thresh:
      count+=1

   if(lst.landmark[9].y*100-lst.landmark[12].y*100)>thresh:
      count+=1  

   if(lst.landmark[13].y*100-lst.landmark[16].y*100)>thresh:
      count+=1

   if(lst.landmark[17].y*100-lst.landmark[20].y*100)>thresh:
      count+=1

   if(lst.landmark[5].x*100-lst.landmark[4].x*100)>thresh:
      count+=1  
    
     
   finger_tip_x=int(lst.landmark[8].x*width) 
   finger_tip_y=int(lst.landmark[8].y*height)

   thumb_tip_x=int(lst.landmark[4].x*width) 
   thumb_tip_y=int(lst.landmark[4].y*height)

   cv2.line(myimage,(finger_tip_x,finger_tip_y),(thumb_tip_x,thumb_tip_y),(255,0,0),2)
    
   center_x=int((finger_tip_x+thumb_tip_x)/2)
   center_y=int((finger_tip_y+thumb_tip_y)/2)
   
   cv2.circle(myimage,(center_x,center_y),2,(0,0,255),2)
   
   distance=math.sqrt(((finger_tip_x- thumb_tip_x)**2)+((finger_tip_y-thumb_tip_y)**2))
   print("what is D ",distance)
    
    # relative mouse x and y
   relative_mouse_x=(center_x/width)*screen_width
   relative_mouse_y=(center_y/height)*screen_height
    
   mymouse.position=(relative_mouse_x,relative_mouse_y) 

   print("what is Mp : ",mymouse.position)

   
   if distance>40:
      if pinch==True:
         pinch=False
         mymouse.release(Button.left)
   if  distance<=40:
      if pinch==False:
         pinch=True
         mymouse.release(Button.left)   

         
   


   totalfingers=count



      
      

   return totalfingers 



while True:
    dummy,image=video.read()
    flipimage=cv2.flip(image,1)

    result=hand_obj.process(cv2.cvtColor(flipimage,cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        hand_keyPoints=result.multi_hand_landmarks[0]
        print(hand_keyPoints)
        count=countFingers(hand_keyPoints,flipimage)
        print("what is fingercount",count)
        cv2.putText(flipimage,"Fingures "+str(count),(200,100),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
        cv2.putText(flipimage,"pinch "+str(pinch),(100,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)



        print(count)
        mydrawing.draw_landmarks(flipimage,hand_keyPoints,myhands.HAND_CONNECTIONS)
    cv2.imshow("handgestures",flipimage)    

    key=cv2.waitKey(1)
    if key==27:
     break

video.release()
cv2.destroyAllWindows()
