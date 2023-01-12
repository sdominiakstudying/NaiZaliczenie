"""
Authors: Stanisław Dominiak s18864, Mateusz Pioch s21331
"""

"""
import the opencv library for the eye recognition, pafy to deal with videos, 
and datetime so that the info on the result looks better
"""
import cv2, pafy, datetime
  
# define a video capture object
user_vid = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# some arbitrarily chosen videos to act like ads
ad_urls = [
    "https://www.youtube.com/watch?v=yGgV7CF8a2Y",
    "https://www.youtube.com/watch?v=ALBioyqdwv0",
    "https://www.youtube.com/watch?v=RLcuIpbNY8s"
]
# videos play 1by1, each time one ends, another one starts
for ad_url in ad_urls:
    ad_video = pafy.new(ad_url)
    ad_best = ad_video.getbest()
    ad_frame_counter = 0
    ad_vid = cv2.VideoCapture(ad_best.url)

    while(True):

        # Capture the video frame from camera
        user_ret, user_frame = user_vid.read()

        # Detect eyes
        user_eyes = eye_cascade.detectMultiScale(user_frame, scaleFactor = 1.2, minNeighbors = 4)

        # Draw a rectangle around eyes
        for (x,y,w,h) in user_eyes:
            cv2.rectangle(user_frame,(x,y),(x+w,y+h),(0, 255, 0),2)
        # Display the resulting frame
        cv2.imshow('You', user_frame)
        
        # If two eyes aren't visible, the video stops. It might go against
        # the spirit of the challenge, but it makes it helluvalot easier
        # to test and debug!
        if len(user_eyes) >= 2:
            #v2.waitKey(5000)
            ad_ret, ad_frame = ad_vid.read()
            ad_frame_counter += 1

            cv2.imshow('Advertisment',ad_frame)
        else:
            # datetime thing makes it easier to tell that new messages are appearing
            print(datetime.datetime.now(), "OPEN YOUR EYES NOW")
            
        # When ad finishes, move to the next one
        if ad_frame_counter == ad_vid.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        # Click q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
# After the loop release the cap objects
user_vid.release()
ad_vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
