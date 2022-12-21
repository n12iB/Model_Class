from argparse import Namespace
import os

class Model():
    def __init__(self,model_type,weights,device_type):
        self.model_type=model_type
        if model_type=="YOLOv7":
            import os
            if not(str(os.getcwd()).endswith("/YOLOv7")):
                os.chdir("./YOLOv7")
            from utils.torch_utils import select_device as YOLO7_select_device
            self.device_type=device_type
            self.device = YOLO7_select_device(device_type)
            self.modelW=self.load_model(weights)
            self.firstRun=True
    
    def load_model(self,weights):
        if self.model_type=="YOLOv7":
            from models.experimental import attempt_load as YOLO7_attempt_load
            modelW = YOLO7_attempt_load(weights, map_location=self.device)
            self.firstRun=True
            return modelW
    
    def predict(self,image,options={}):
        if self.model_type=="YOLOv7":
              
            from detect import detect as YOLO7_detect
            opt = Namespace(agnostic_nms=False, 
                            augment=False, 
                            classes=None, 
                            conf_thres=0.25, 
                            device=self.device_type, 
                            exist_ok=False, 
                            img_size=640, 
                            iou_thres=0.45, 
                            name='exp', 
                            no_trace=not(self.firstRun), 
                            nosave=False, 
                            project='runs/detect', 
                            save_conf=True, 
                            save_txt=True,
                            return_txt=True, 
                            source=image, 
                            update=False, 
                            view_img=False)
            opt_dict=vars(opt)
            for i in options:
                opt_dict[i]=options[i]

            self.firstRun=False 
            values=YOLO7_detect(opt,self.modelW)
            return values