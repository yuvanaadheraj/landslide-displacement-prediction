# src/config.py
class Config:
    DATA_PATH = "datalandslide.csv"  # <-- FIX 1
    EPOCHS = 50
    LR = 0.001
    HIDDEN_DIM = 64
    OUTPUT_DIM = 3  # <-- FIX 2 (Must be 3 for dispx, dispy, dispz)
    BATCH_SIZE = 32
    TRAIN_SPLIT = 0.8
    
    # New setting for GNN
    SEQ_LEN = 7  # Use 7 days of data to predict the next day