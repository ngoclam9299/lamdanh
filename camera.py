#from pyimagesearch.shapedetector import ShapeDetector
import numpy as np
import cv2
import argparse
import imutils

# đọc webcam
cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    # resize ảnh
    image = cv2.resize(frame,(320,240))
   
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        cv2.drawContours(frame,[approx],0,(0,0,0),5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx)== 3:
            cv2.putText(image,"Triangle",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
        elif len(approx)== 4:
            cv2.putText(image,"Rectangle",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))


    # chuyển dổi hệ màu từ BGR sang HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # mã màu quy địnhq
    min_mau_y = np.array([15,111,144]) #yellow
    max_mau_y = np.array([18,123,155]) 
    min_mau_r = np.array([173,138,123]) #red
    max_mau_r = np.array([176,176,157])
    min_mau_b = np.array([100,50,145]) #blue
    max_mau_b = np.array([120,215,255])
    
    # mặt nạ từng lớp màu
    mask_y = cv2.inRange(hsv,min_mau_y,max_mau_y)
    mask_r = cv2.inRange(hsv,min_mau_r,max_mau_r)
    mask_b = cv2.inRange(hsv,min_mau_b,max_mau_b)
   
    # kết quả so sánh với mặt nạ màu
    final_y = cv2.bitwise_and(image,image,mask=mask_y)
    final_r = cv2.bitwise_and(image,image,mask=mask_r)
    final_b = cv2.bitwise_and(image,image,mask=mask_b)
    
    # so sanh dieu kien mau
    if np.any(final_y): #màu vàng
        print("màu vàng")
        cv2.imshow('ảnh màu vàng',final_y)

    elif np.any(final_r): #màu đỏ
        print("màu đỏ")
        cv2.imshow('ảnh màu đỏ',final_r)

    elif np.any(final_b): #màu xanh
        print("màu xanh")
        cv2.imshow('ảnh màu xanh',final_b)

    else: #không có màu RGB
        print("không có màu RGB")
        
    cv2.imshow('ảnh gốc',image)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()