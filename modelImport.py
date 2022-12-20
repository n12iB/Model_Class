from YOLOv7.models.experimental import attempt_load as YOLO7_attempt_load
from YOLOv7.utils.torch_utils import select_device as YOLO7_select_device
from YOLOv7.detect import detect as YOLO7_detect
from argparse import Namespace


class Model():
    def __init__(self,model_type,weights,device_type):
        self.model_type=model_type
        if model_type=="YOLOv7":
            self.device = YOLO7_select_device(device_type)
            self.model=self.load_model(weights)

    
    def load_model(self,weights):
        if self.model_type=="YOLOv7":
            model = YOLO7_attempt_load(weights, map_location=self.device)
            return model
    
    def predict(self,image,options={}):
        if self.model_type=="YOLOv7":   
            opt = Namespace(agnostic_nms=False, 
                            augment=False, 
                            classes=None, 
                            conf_thres=0.25, 
                            device='', 
                            exist_ok=False, 
                            img_size=640, 
                            iou_thres=0.45, 
                            name='exp', 
                            no_trace=False, 
                            nosave=False, 
                            project='runs/detect', 
                            save_conf=True, 
                            save_txt=True,
                            return_txt=True, 
                            source=image, 
                            update=False, 
                            view_img=False)
            
            return YOLO7_detect(opt,self.model)