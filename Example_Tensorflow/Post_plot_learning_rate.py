import matplotlib.pyplot as plt

def plot_hist(history,metric_str='accuracy'):
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc=history[metric_str]
    val_acc=history['val_'+metric_str]
    loss=history['loss']
    val_loss=history['val_loss']
    
    epochs=range(len(acc)) # Get number of epochs
    
    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot(epochs, acc, 'r', label="Training "+metric_str)
    plt.plot(epochs, val_acc, 'b', label="Validation "+metric_str)
    plt.title('Training and validation '+metric_str)
    plt.xlabel('epochs')
    plt.ylabel(metric_str)
    plt.legend()
    plt.show()
    print("")
    
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show() 