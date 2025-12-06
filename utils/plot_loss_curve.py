import matplotlib.pyplot as plt


def plot_loss_curve(train_loss_count, test_loss_count):
    plt.figure(figsize = (7,4))
    plt.plot(train_loss_count, label = 'train loss')
    plt.plot(test_loss_count, label = 'test loss')
    plt.legend()
    plt.title('loss curve')
    plt.show()