import matplotlib.pyplot as plt



def visualize_preds(batch_images, predictions):
    n = 10
    plt.figure(figsize=(20,5))
    for i in range(n):
        
        ax = plt.subplot(2, n, i+1)
        ax.set_title("Original Image")
        plt.imshow(batch_images[i].reshape(256,256,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i+1+n)
        ax.set_title("Predicted Image")
        plt.imshow(predictions[i].reshape(256,256,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)



