import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Cross-entropy loss
        cross_entropy = -y_true * K.log(y_pred)
        
        # Modulating factor: (1 - p_t)^gamma
        weight = alpha * K.pow((1 - y_pred), gamma)
        
        # Focal loss
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    
    return focal_loss_fixed

# Usage in model compilation
model.compile(optimizer='adam', loss=focal_loss(gamma=2., alpha=0.25))
