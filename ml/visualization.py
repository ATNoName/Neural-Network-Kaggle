import matplotlib.pyplot as plt

def plot_scatter(x, y, marker = 4, color = 'blue', title = ' ', x_label = ' ', y_label = ' '):
    for i in range(x.shape[1]):
        plt.figure(i)
        plt.scatter(x[:, i], y, marker, color)
        plt.title(title)
        plt.xlabel(x_label + ' ' + str(i))
        plt.ylabel(y_label)
    plt.show()
    
def plot_histo(x, bins = 10, color = 'blue', title = ' ', x_label = ' '):
    for i in range(x.shape[1]):
        plt.figure(i)
        plt.hist(x[:, i], bins, color = color)
        plt.title(title)
        plt.xlabel(x_label + ' ' + str(i))
        plt.ylabel('Frequency')
    plt.show()
        
def plot_nn_loss(nn):
    plt.plot(nn.epoch, nn.loss, 'Loss vs Epoch', 'Epoch', 'Loss')
    
def show_image(image, prediction, actual):
    plt.imshow(image)
    plt.annotate('Prediction: ' + prediction + '\nActual: ' + actual, xy=(0, 0), xytext=(0, -1))
    plt.show()