# Robust Knowledge Distillation Based on Feature Variance for Backdoored Teacher Model
Implementation of the paper "Robust Knowledge Distillation Based on Feature Variance for Backdoored Teacher Model".
## Setup
  1.conda env create -f environment.yml # Creates Anaconda env with requirements  
  2.git clone (https://github.com/Xming-Z/RobustKD.git) # Download RobustKD repository  
## Train the normal teacher model
python train_teacher_poison.py  
## Create the patch
python create_patch.py  
## Train the poisoned teacher model
python train_teacher_poison.py # Loading trained patches to generate poisoned teacher models  
## Robust Knowledge Distillation
python train_with_distillation.py
