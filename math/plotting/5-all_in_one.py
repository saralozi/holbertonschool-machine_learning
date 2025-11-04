#!/usr/bin/env python3
"""A script that plots 5 graphs in one figure"""


import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """A function that plots 5 graphs in 1 figure"""

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle('All in One')

    axs[0, 0].plot(y0, color='red')
    axs[0, 0].set_xlim(0, 10)
    axs[0, 0].set_ylim(0, 1000)
    axs[0, 0].set_yticks(np.arange(0, 1001, step=500))

    axs[0, 1].scatter(x1, y1, color='magenta')
    axs[0, 1].set_title("Men's Height vs Weight", fontsize='x-small')
    axs[0, 1].set_xlabel("Height (in)", fontsize='x-small')
    axs[0, 1].set_ylabel("Weight (lbs)", fontsize='x-small')
    axs[0, 1].set_xlim(55, 83)
    axs[0, 1].set_ylim(165, 195)
    axs[0, 1].set_xticks(np.arange(60, 81, step=10))
    axs[0, 1].set_yticks(np.arange(170, 191, step=10))

    axs[1, 0].plot(x2, y2)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title("Exponential Decay of C-14", fontsize='x-small')
    axs[1, 0].set_xlabel("Time (years)", fontsize='x-small')
    axs[1, 0].set_ylabel("Fraction Remaining", fontsize='x-small')
    axs[1, 0].set_xlim(x2[0], x2[-1])
    axs[1, 0].set_xticks(np.arange(x2[0], x2[-1], step=10000))

    axs[1, 1].plot(x3, y31, color='red', linestyle='dashed', label='C-14')
    axs[1, 1].plot(x3, y32, color='green', label='Ra-226')
    axs[1, 1].set_title(
        "Exponential Decay of Radioactive Elements", fontsize='x-small')
    axs[1, 1].set_xlabel("Time (years)", fontsize='x-small')
    axs[1, 1].set_ylabel("Fraction Remaining", fontsize='x-small')
    axs[1, 1].set_xlim(0, 20001)
    axs[1, 1].set_ylim(0, 1.01)
    axs[1, 1].set_xticks(np.arange(0, 20001, step=5000))
    axs[1, 1].set_yticks(np.arange(0, 1.01, step=0.5))
    axs[1, 1].legend(fontsize='x-small')

    gs = axs[2, 0].get_gridspec()
    axs[2, 0].remove()
    axs[2, 1].remove()
    ax_hist = fig.add_subplot(gs[2, :])

    ax_hist.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')
    ax_hist.set_title("Project A", fontsize='x-small')
    ax_hist.set_xlabel("Grades", fontsize='x-small')
    ax_hist.set_ylabel("Number of Students", fontsize='x-small')
    ax_hist.set_xlim(0, 100)
    ax_hist.set_ylim(0, 30)
    ax_hist.set_yticks(np.arange(0, 31, step=10))
    ax_hist.set_xticks(np.arange(0, 101, step=10))

    plt.tight_layout()
    plt.show()
