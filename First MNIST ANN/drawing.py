import cv2
import numpy as np
import digit_recognition as dr

# creating a 600 x 600 pixels canvas for mouse drawing
canvas = np.ones((600,600), dtype="uint8") * 255
# designating a 400 x 400 pixels point of interest on which digits will be drawn
canvas[100:500,100:500] = 0

start_point = None
end_point = None
is_drawing = False

def draw_line(img,start_at,end_at):
    cv2.line(img,start_at,end_at,255,50)

def on_mouse_events(event,x,y,flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_drawing:
            start_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x,y)
            draw_line(canvas,start_point,end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False


cv2.namedWindow("Test Canvas")
cv2.setMouseCallback("Test Canvas", on_mouse_events)


while(True):
    cv2.imshow("Test Canvas", canvas)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_drawing = not is_drawing
    elif key == ord('c'):
        canvas[100:500,100:500] = 0
    elif key == ord('p'):
        image = canvas[100:500,100:500]
        input = cv2.resize(image, (20 , 20)).reshape((20 , 20,1)).astype('float32') / 255
        input = np.array(input).reshape(1,-1)
        prediction = dr.predict(input, dr.optimized_theta1, dr.optimized_theta2)
        print(f"The prediction for this digit is: {prediction}")

        


    
    

