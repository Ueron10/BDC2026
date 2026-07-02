import os


class Config:
    DATA_DIR = 'data'
    MODEL_DIR = 'model'
    OUTPUT_DIR = 'output'
    DOCS_DIR = 'output'
    TEMPLATE_PATH = 'output/submission.csv'
    OUTPUT_PATH = 'output/submission.csv'
    
    NUM_CLASSES = 3
    CLASS_NAMES = ['Recyclable', 'Electronic', 'Organic']
    CLASS_TO_IDX = {'0_Recycleable': 0, '1_Electronic': 1, '2_Organic': 2}
    
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    VAL_SPLIT = 0.15
    SEED = 42
    
    PRETRAINED = True
    
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    USE_MIXUP = True
    USE_TTA = True
    TTA_TRANSFORMS = 5
    
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    EARLY_STOPPING_PATIENCE = 15
    
    @staticmethod
    def get_project_root():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @staticmethod
    def get_absolute_path(relative_path):
        return os.path.abspath(os.path.join(Config.get_project_root(), relative_path))
