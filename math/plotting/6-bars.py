#!/usr/bin/env python3
"""A script that plots a stacked bar graph"""


import numpy as np
import matplotlib.pyplot as plt


def bars():
    """A function that plots a stacked bar graph"""

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(fruit.shape[1])
    width = 0.5

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(x, apples, width, label='apples', color='red', edgecolor='none')
    plt.bar(x, bananas, width, bottom=apples, label='bananas',
            color='yellow', edgecolor='none')
    plt.bar(x, oranges, width, bottom=apples + bananas, label='oranges',
            color='#ff8000', edgecolor='none')
    plt.bar(x, peaches, width, bottom=apples + bananas + oranges,
            label='peaches', color='#ffe5b4', edgecolor='none')

    plt.xticks(x, people)
    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.title('Number of Fruit per Person')

    plt.legend(loc='upper right', frameon=True)

    plt.show()
