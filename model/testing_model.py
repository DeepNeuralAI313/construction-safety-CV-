import cv2
from ultralytics import YOLO


def video(model_path,video_path):
    model=YOLO(model_path)
    cap=cv2.VideoCapture(video_path)
    while True:
        _,img=cap.read()
        result=model(img)
        cv2.imshow('respons',result[0].plot())

        if cv2.waitKey(2) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            

def image(model_path,image_path):
    model=YOLO(model_path)
    img=cv2.imread(image_path)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=model(img)   
    cv2.imshow('respons',result[0].plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
if __name__ == "__main__":
    model_path='model/healmet_detaction.pt'
    video_path=0
    image_path='model/test_images/00404.jpg'
    if image_path!='':
        image(model_path,image_path)
    else :
        video(model_path,video_path)