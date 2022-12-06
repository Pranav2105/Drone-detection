from tkinter import *
from PIL import Image,ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pygame import mixer
mixer.init()
sound = mixer.Sound('audio.wav')

net = cv2.dnn.readNetFromDarknet(r"C:\Users\ayapi\Downloads\drone-net-master\drone-net-master\yolo-drone.cfg",r"C:\Users\ayapi\Downloads\drone-net-master\drone-net-master\yolo-drone.weights")
class Drone_Detection:
    def __init__(self,root):
        self.root = root
        self.root.title("Drone Detection")
        self.root.geometry("800x500")

        label = Label(self.root,text="Drone Detection",font=("times new roman",20,"bold"))
        label.pack(fill=X)

        img = Image.open("./bg.jpg").resize((800,455))
        img = ImageTk.PhotoImage(img)
        
        bg = Label(self.root,image=img)
        bg.image = img
        bg.place(x=0,y=40,width=800,height=455)

        button = Button(self.root,text="Detect",font=("times new roman",16,"bold"),bd=4,relief=RIDGE,command=self.detect)
        button.place(x=360,y=400)

    def detect(self):
        classes = ['drone']
        cap = cv2.VideoCapture(0)

        while 1:
            _, img = cap.read()
            img = cv2.resize(img,(1280,720))
            hight,width,_ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

            net.setInput(blob)

            output_layers_name = net.getUnconnectedOutLayersNames()

            layerOutputs = net.forward(output_layers_name)

            boxes =[]
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]
                    if confidence > 0.7:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * hight)
                        w = int(detection[2] * width)
                        h = int(detection[3]* hight)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)


            indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

            boxes =[]
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]
                    if confidence > 0.5:
                        sound.play()
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * hight)
                        w = int(detection[2] * width)
                        h = int(detection[3]* hight)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)



                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
            print(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0,255,size =(len(boxes),3))
            if  len(indexes)>0:
                for i in indexes.flatten():
                    x,y,w,h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                    cv2.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)

            cv2.imshow('img',img)
            if cv2.waitKey(1) == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()







if __name__ == "__main__":
    root = Tk()
    Drone_Detection(root)
    root.mainloop()


