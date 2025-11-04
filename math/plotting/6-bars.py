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
    apples, bananas, oranges, peaches = fruit

    x = np.arange(fruit.shape[1])
    width = 0.5

    plt.bar(
        x,
        apples,
        width,
        label='apples',
        color='red')
    plt.bar(
        x,
        bananas,
        width,
        bottom=apples,
        label='bananas',
        color='yellow')
    plt.bar(
        x,
        oranges,
        width,
        bottom=apples + bananas,
        label='oranges',
        color='#ff8000')
    plt.bar(
        x,
        peaches,
        width,
        bottom=apples + bananas + oranges,
        label='peaches',
        color='#ffe5b4')

    plt.xticks(x, people)
    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.tight_layout()
    plt.show()
