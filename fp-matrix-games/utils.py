import numpy as np
import matplotlib.pyplot as plt

def make_plot(data):
    f,(ax1,ax2)=plt.subplots(1,2,figsize=(16,8),dpi=220, facecolor='w', edgecolor='k')
    ax1.plot(data[:,0],label="first player")
    ax1.plot(data[:,1],label="second player")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("strategy")
    ax1.legend(loc="upper right", bbox_to_anchor=(1,-0.05))  
    
    ax2.plot(data[:,0],data[:,1],'.--',label="strategy profile")
    ax2.annotate("", xy=(data[1,0], data[1,1]), xytext=(data[0,0], data[0,1]),arrowprops=dict(arrowstyle="fancy"))
    ax2.set_xlabel("first player's strategy")
    ax2.set_ylabel("second player's strategy")
    ax2.legend()
    plt.show()        