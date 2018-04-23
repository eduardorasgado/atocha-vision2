
#Color filter Line Detector
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import urllib.request as urllib

width = 600
height =500


url = "http://192.168.0.11:8080/shot.jpg"

def roi(img,width,height):
    #vertices = np.array([[0,500],[10,450],[200,300],[400,300],[540,450],[600,500]],np.int32)
    #vertices = np.array([[0,width],[0,400],[300,200],[width,400],[width,height]],np.int32)
    vertices = np.array([[160,width],[160,400],[280,330],[320,330],[width-160,400],[width-160,height]],np.int32)
    vertices = [vertices]
    #blank mask
    mask = np.zeros_like(img)
    #fill the mask
    cv2.fillPoly(mask,vertices,255)
    #show the area that is the mask
    masked = cv2.bitwise_and(img,mask)
    return masked

def roi2(img,width,height):
    #vertices = np.array([[0,500],[10,450],[200,300],[400,300],[540,450],[600,500]],np.int32)
    vertices = np.array([[100,width],[100,400],[300,300],[width/2,height]],np.int32)
    vertices = [vertices]
    #blank mask
    mask = np.zeros_like(img)
    #fill the mask
    cv2.fillPoly(mask,vertices,255)
    #show the area that is the mask
    masked = cv2.bitwise_and(img,mask)
    return masked
def roi3(img,width,height):
    #vertices = np.array([[0,500],[10,450],[200,300],[400,300],[540,450],[600,500]],np.int32)
    vertices = np.array([[width/2,height],[300,300],[width-100,400],[width-100,height]],np.int32)
    vertices = [vertices]
    #blank mask
    mask = np.zeros_like(img)
    #fill the mask
    cv2.fillPoly(mask,vertices,255)
    #show the area that is the mask
    masked = cv2.bitwise_and(img,mask)
    return masked

def roi_car_det(img,width,height):
    vertices = np.array([[0,500],[10,450],[150,200],[450,200],[540,450],[600,500]],np.int32)
    #vertices = np.array([[0,width],[0,400],[300,200],[width,400],[width,height]],np.int32)
    #vertices = np.array([[160,width],[160,400],[280,330],[320,330],[width-160,400],[width-160,height]],np.int32)
    vertices = [vertices]
    #blank mask
    mask = np.zeros_like(img)
    #fill the mask
    cv2.fillPoly(mask,vertices,255)
    #show the area that is the mask
    masked = cv2.bitwise_and(img,mask)
    return masked

def draw_lines1(lines,processed_imgen):
    coords = 0
    try:
        for line in lines:
            coords = line[0]
            cv2.line(processed_imgen,(coords[0],coords[1]),(coords[2],coords[3]),[0,255,0],10)
            #cv2.line(processed_imgen,(coords[0],coords[1]),(coords[2],coords[3]),[0,255,0],10)
            #print(coords)
            return processed_imgen, coords
    except Exception as e:
        pass
    return processed_imgen, coords

def car_detector(frame,gray,car_cascade):
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
     
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    return frame

def obtain_pixels(frame,width,height,last_px):
    higher = 0
    maskd = roi(frame,width,height)
    for i in range(len(maskd)):
        if i >175:
            region =maskd[i,380]
            #linea de filtro de color
            #cv2.circle(maskd,(i,380),5,(120,50,200),-1)
            if region !=0:
                #print(region)
                if region>higher:
                    higher=region      
    if higher == 0:
        higher = last_px-47
        
    average_pix =higher+47
    #print(average_pix)
    return average_pix,maskd

#recordar limpiar el error integrativo
past_diffpx=0
sigma_px=0
last_pixels =0
#lectura del xml
#car_cascade = cv2.CascadeClassifier('cars.xml')

#cap = cv2.VideoCapture(0)
while True:
    if sigma_px>10000000:
        sigma_px=0
    #taking video capture
    #_,frame = cap.read()
    imgResp = urllib.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame = cv2.imdecode(imgNp,-1)
    #frame = grab_screen(region=(10,70,width,height))
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    #transforming into grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #frame.set(3,width)
    #frame.set(4,height)
    #Car detector
    #gray_mod = roi_car_det(gray,width,height)
    #frames = car_detector(frame,gray_mod,car_cascade)
    
    #PID CONTROL FOR THRESHOLD
    pixels,makk = obtain_pixels(gray,width,height,last_pixels)
    diff_px=pixels-last_pixels
    #try the best p,d,i gains
    pix_fix = 0.2*diff_px+1.1*(diff_px-past_diffpx)+0.004*sigma_px
    #pix_fix = 6.0*diff_px+6.0*((diff_px-past_diffpx)/0.26)+0.0004*sigma_px
    past_diffpx=diff_px                       
    last_pixels+=pix_fix                        
    ret, gray = cv2.threshold(gray,last_pixels,255,cv2.THRESH_BINARY)#140,255 #last pixel=low_threshold
    print(last_pixels)
    print("adjustment: ",pix_fix)
    last_pixels=pixels
    sigma_px +=diff_px
    """
    #############This is for white and yellow differentiation
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #RGB2HSV
    lower_yellow = np.array([22,60,20])
    upper_yellow = np.array([60,255,255])
    #loweryellow = np.array([0])
    #upperyellow = np.array([25])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel0 = np.ones((15,15), np.uint8)
    opening= cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel0)
    mask=opening
    result1 = cv2.bitwise_and(frame,frame,mask=mask)
    kernel = np.ones((2,2), np.uint8)
    result1 = cv2.dilate(result1,kernel,iterations=2)
    ###################
    """
  
    #lower_white = np.array([12])#125,134
    #upper_white = np.array([255])

    #mask = cv2.inRange(gray, lower_white, upper_white)
  
    kernel = np.ones((2,2), np.uint8)
    #eliminar fals0s positivos
    #mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    processed = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
    #so esta en ense rango cada arreglo regresa un True
    #processed = cv2.bitwise_and(frame,frame,mask=mask)
    
    #si existe un 1 en la mask se mstrara color en el frame, 
    #si existe un o se mostrara no color
    
    
    #processed = cv2.GaussianBlur(processed,(5,5),3) #image,kernel,blur grade
    canny_white = cv2.Canny(processed,150,300)
    canny_white2 = cv2.Canny(processed,150,300)
    canny_white = cv2.dilate(canny_white,kernel,iterations=2)#1
    canny_white2 = cv2.dilate(canny_white2,kernel,iterations=2)#1
    #processed = roi(processed,width,height)
    #############
    
    processed_imgen = roi2(canny_white,width,height)
    processed_imgen2 = roi3(canny_white2,width,height)
    
    img2, contours, hier = cv2.findContours(processed_imgen,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img3, cont, hier2 = cv2.findContours(processed_imgen2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cnt = cont[0]
    #cv2.drawContours(frame,[cnt],-1,(0,0,0),3)
    #cv2.drawContours(frame,contours,-1,(0,255,0),-1)#3 o -1
    list_x1 = []
    list_y1 = []
    list_x2 = []
    list_y2 = []
    
    conto = {}
    tolerance = 3
    try:
        for k in range(len(contours)):
            cnt = len(contours[k])
            conto[cnt] = contours[k]
            #print(contours[k])
        #print(cont)
        rev = []
        for key in conto:
            #print(key)
            rev.append(int(key))
        rev = sorted(rev)
        rev.reverse()
        #print(rev)   
        for key in conto:      
            for k in range(3):              
                if rev[k] == key:
                    compr_list = []
                    counter = 0
                    cc = 0
                    for n in range(len(conto[key])):
                        if cc>0:
                            line_compr = abs(conto[key][n][0][0]-conto[key][n-1][0][0])
                            compr_list.append(line_compr)                                                
                        #print(compr_list)
                        cc+=1
                    for j in range(len(compr_list)):
                        if compr_list[j] > tolerance:
                            counter+=1
                    if counter > 6:
                        continue
                    elif counter <=6:
                        cv2.drawContours(frame,[conto[key]],-1,(255,0,0),6)
                        for ar in range(len(conto[key])):
                            list_x1.append(conto[key][ar][0][0])
                            list_y1.append(conto[key][ar][0][1])
    except Exception as e:
        #print(e,"line: ",sys.exc_info()[2].tb_lineno)
        pass
                    
    conto2 = {}
    try:
        for k in range(len(cont)):
            cnt = len(cont[k])
            conto2[cnt] = cont[k]
            #print(contours[k])
        #print(cont)
        rev2 = []
        for key in conto2:
            #print(key)
            rev2.append(int(key))
        rev2 = sorted(rev2)
        rev2.reverse()
        #print(rev2)   
        for key in conto2:      
            for k in range(2):              
                if rev2[k] == key:
                    compr_list = []
                    counter = 0
                    cc = 0
                    for n in range(len(conto2[key])):
                        if cc>0:
                            line_compr = abs(conto2[key][n][0][0]-conto2[key][n-1][0][0])
                            compr_list.append(line_compr)                                                
                        #print(compr_list)
                        cc+=1
                    for j in range(len(compr_list)):
                        if compr_list[j] > tolerance:
                            counter+=1
                    if counter > 6:
                        continue
                    elif counter <=6:
                        cv2.drawContours(frame,[conto2[key]],-1,(0,0,255),6)
                        for ar in range(len(conto2[key])):
                            list_x2.append(conto2[key][ar][0][0])
                            list_y2.append(conto2[key][ar][0][1])

                    
    except Exception as e:
        #print(e,"line: ",sys.exc_info()[2].tb_lineno)
        pass   
    try:
        #for i in range(len(list_x1)):
        #    cv2.circle(frame,(list_x1[i]+5,list_y1[i]+5),5,(130,50,200),-1)
        this=0
        #num = random.randint(0,len(list_x2)-1)
        num =len(list_x2)/2
        midd = list_x2[num]
        for i in range(len(list_x1)):
            if list_x1[i]==midd:
                this = i
        pre = (midd-list_x1[this])/2
        averagex =list_x1[num]+pre  #altertative: replace num by this, not good at all
        averagey = list_y2[num]
        if averagex > list_x2[num] and averagey > max(list_y1) and averagex<list_x1[this]:
            continue
        else:
            cv2.circle(frame,(averagex,averagey),5,(130,50,200),-1)
            cv2.line(frame,(averagex,averagey+200),(averagex,averagey-100),[255,255,255],2)
            cv2.putText(frame,"middle road={0},{1}".format(averagex,averagey),(averagex+5,averagey+5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0))
    except Exception as e:
        pass
    
    indice_for_cnt = 0
    indice_for_cnt2 =0
    longer = []
    try:
        for k in range(len(contours)):
            cnt = len(contours[k])
            longer.append(cnt)
        maxim = max(longer)
        
        for i in range(len(longer)):
            if longer[i] == maxim:
                indice_for_cnt = i
        first_line = cv2.drawContours(frame,[contours[indice_for_cnt]],-1,(255,0,0),6)
    except Exception as e:
        #print(str(e))
        pass
    
    longer2 = []
    try:
        for k in range(len(cont)):
            cnt = len(cont[k])
            longer2.append(cnt)
        maxim2 = max(longer2)
        for i in range(len(longer2)):
            if longer2[i]==maxim2:
                indice_for_cnt2 = i
        #the longest array of contour is the seleected for draw
        second_line = cv2.drawContours(frame,[cont[indice_for_cnt2]],-1,(0,0,255),6)
    except Exception as e:
        #print("here",e)
        pass
    
    """
    xs1 = []
    ys1 = []
    xs2 = []
    ys2 = []
    try:
        #print(contours[indice_for_cnt][0][0][0])
        
        mid_cont = len(contours[indice_for_cnt])/2
        total_cont = len(contours[indice_for_cnt])-1 #(mid_cont/2)
        for i in range(mid_cont,total_cont):
            xs1.append(contours[indice_for_cnt][i][0][0])
            ys1.append(contours[indice_for_cnt][i][0][1])
        xs1.reverse()
        ys1.reverse()
        mid_cont2 = len(cont[indice_for_cnt2])/3
        total_cont2 = len(cont[indice_for_cnt2])-(mid_cont2/3)
        for i in range(mid_cont2,total_cont2):
            xs2.append(cont[indice_for_cnt2][i][0][0])
            ys2.append(cont[indice_for_cnt2][i][0][1])
        mid1 = len(xs1)/2
        mid2 = len(xs2)/2
                  
        shape = np.array([[xs2[0],xs2[0]],[xs2[mid2],ys2[mid2]],[xs2[-1],xs2[-1]],
                          [xs1[0],ys1[0]],[xs1[mid1],ys1[mid1]],[xs1[-1],ys1[-1]]],np.int32)
        shape = [shape]
        cv2.fillPoly(frame,shape,(255,255,255))
    except Exception as e:
        print(str(e))
        pass
    """                    
   
    #cv2.imshow("White lines", processed_imgen)
    #cv2.imshow("White lines2", processed_imgen2)
    cv2.imshow("prs", processed)
    cv2.imshow("constrain for PID", makk)
    #cv2.imshow("canny", canny_white)
    #cv2.imshow("gray", gray)
    #cv2.imshow("amsk", processed)
    #cv2.imshow("car detection",gray_mod)
    #cv2.imshow("original",cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    cv2.imshow("SDC_Percetion: Atocha.ai",frame)
    k = cv2.waitKey(5) 
    if k & 0xFF ==ord('q'):
        break
#cap.release()
cv2.destroyAllWindows()