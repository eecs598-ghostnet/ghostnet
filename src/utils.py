
import matplotlib.pyplot as plt

def show_plot(iteration,loss):
    fig = plt.figure()
    plt.plot(iteration,loss)
    fig.savefig('loss.png')
