# Subjective and Objective Audio-Visual Quality Assessment for User Generated Content

## Description
ANNAVQA code for the following papers:

- Y. Cao, X. Min, W. Sun, and G. Zhai, "Subjective and objective audio-
visual quality assessment for user generated content," IEEE Transactions
on Image Processing, 2023.
- Y. Cao, X. Min, W. Sun, X. Zhang and G. Zhai, "Audio-Visual Quality Assessment for User Generated Content: Database and Method," IEEE International Conference on Image Processing, 2023, pp. 1495-1499.

## SJTU-UAV Dataset
Dataset Link: http://automation.sjtu.edu.cn/en/ShowPeople.aspx?info_id=366&info_lb=326&flag=224


## GeneralAVQA

### Training on SJTU-UAV Dataset
1. Download the SJTU-UAV Database 


2. Extract audio features
    ```
    python extract_audio_feature.py --features_dir=<path to save audio features> --videos_dir==<SJTU-UAV database path>
    ```

3. Extract video features
    Before extracting video features, you should first download pre-trained weights from https://github.com/zwx8981/TCSVT-2022-BVQA and put it under SpatialExtractor/weights. 
    ```
    python extract_spatial_feature.py --features_dir=<path to save video features> --videos_dir==<SJTU-UAV database path>
    ```
4. Train model
   pytorch version
    ```
    python train_pytorch.py --videofeatures_dir=<path where save video features> --audiofeatures_dir=<path where save audio features> --videos_dir==<SJTU-UAV database path>
    ```
   mindscope version
    ```
    python train_ms.py --videofeatures_dir=<path where save video features> --audiofeatures_dir=<path where save audio features> --videos_dir==<SJTU-UAV database path>
    ```
    
