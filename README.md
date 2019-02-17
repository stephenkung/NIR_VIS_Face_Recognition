# NIR_VIS_Face_Recognition
NIR-VIS face recognition for CASIA NIR-VIS2.0 dataset, implemented by Pytorch. Scripts only for testing and reporting performance.   

## description
1) extract_features.py: apply pretrained deep learning model(LightCNN 9/29) to extract face embedding(128D).            
Compute Rank1 accuracy and AR@FAR=0.01.  It implements the view2 protocal defined by CASIA2.0 dataset.       
2) light_cnn.py: the standard LightCNN model.     
Acess here for more about LightCNN: https://github.com/AlfredXiangWu/LightCNN    

## prerequisite
1) You need to do face detection and face alignment, then crop face images to 112*112.    
2) Download pretrained LightCNN weight.   

