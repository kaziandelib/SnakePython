"""
This module provides functionality for visualizing the training progress of a reinforcement learning agent 
using the matplotlib library. It defines an interactive plotting function that updates in real-time to display 
individual game scores and their mean scores as the training progresses. The function manages output in Jupyter 
notebooks by clearing the previous plots, allowing for a dynamic and uncluttered visual representation of the 
agent's performance over time.
"""


import matplotlib.pyplot as plt  # To import the matplotlib library for creating visualizations.
from IPython import display  # To import the display module from IPython for output management in Jupyter notebooks.

plt.ion()  # To enable interactive mode in matplotlib, allowing for dynamic updates to the plot.

# To define a function that plots scores and mean scores during training.
def plot(scores, mean_scores):
    display.clear_output(wait=True)  # To clear the current output in the notebook, preventing clutter during updates.
    display.display(plt.gcf())  # To display the current figure (plot) in the notebook.
    plt.clf()  # To clear the current figure, preparing it for the next plot.

    # To set the title of the plot indicating the ongoing training process.
    plt.title('Training...')
    plt.xlabel('Number of Games')  # To label the x-axis with 'Number of Games'.
    plt.ylabel('Score')  # To label the y-axis with 'Score'.

    # To plot the scores from individual games with a label for the legend.
    plt.plot(scores, label='Scores')  
    # To plot the mean scores with a label for the legend.
    plt.plot(mean_scores, label='Mean Scores')  

    plt.ylim(ymin=0)  # To set the minimum limit of the y-axis to 0, ensuring all scores are visible.

    # To annotate the last score on the plot with its value.
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    # To annotate the last mean score on the plot with its value.
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # To add a legend in the top left corner to distinguish between individual scores and mean scores.
    plt.legend(loc='upper left')
    
    # To display the plot without blocking the execution of subsequent code.
    plt.show(block=False)
    # To pause for a brief moment to allow the plot to render before the next update.  
    plt.pause(.1)  
