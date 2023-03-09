# Library
import cv2, time, pandas
from datetime import datetime
  
# Assign "static_back" to None
static_back = None
  
# List to store any moving object
motion_list = [None, None]
  
# Time
time = []
  
# DataFrame (start time and end time)
df = pandas.DataFrame(columns = ["Start", "End"])
  
# Video Capture
video = cv2.VideoCapture(0)
  
# Infinite loop while the video is still active
while True:
    check, frame = video.read()
    motion = 0
  
    # image to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # grayscale image to GaussianBlur 
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
  
    if static_back is None:
        static_back = gray
        continue
  
    # Difference between static background and GaussianBlur
    diff_frame = cv2.absdiff(static_back, gray)
  
    # If the change between the static background and the current frame is more than 30, it will display white (255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
  
    # Contour of a moving object
    cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1
  
        (x, y, w, h) = cv2.boundingRect(contour)
        # Red rectangle on the object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(frame, "ALERT !!", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
  
    motion_list.append(motion)
    motion_list = motion_list[-2:]
  
    # Appending Start time of motion
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())
  
    # Appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())
  
    # Output (Display)
    cv2.imshow("Grayscale Image", gray)  
    cv2.imshow("Difference Frame", diff_frame)  
    cv2.imshow("Threshold Frame", thresh_frame)  
    cv2.imshow("Color Frame", frame)
  
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Something is moving then it append the end time of movement
        if motion == 1:
            time.append(datetime.now())
        break
  
# Appending in DataFrame
for i in range(0, len(time), 2):
    df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True)
  
df.to_csv("Time_of_movements.csv")
video.release()  
cv2.destroyAllWindows()