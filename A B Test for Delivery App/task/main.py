# write your code here
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import mannwhitneyu
from statsmodels.stats.power import TTestIndPower

"""
Levene' s test.
stat: Test statistics, variation between values
p-value: Greater than 0.05, then the variances are not significantly different between groups.

T-test.
Parametric test used to test for a statistically significant difference in the means between groups.
equal_var=True: The standard independent two sample t-test will be conducted by taking into consideration the equal population variances.
"""


def stage1():
    df = pd.read_csv('aa_test.csv')
    sample1 = list(df["Sample 1"])
    sample2 = list(df["Sample 2"])

    stat, p = st.levene(sample1, sample2, center='mean')
    p_test = "p-value > 0.05" if p > 0.05 else "p-value <= 0.05"
    vt1, vt2 = ("no", "yes") if p > 0.05 else ("yes", "no")

    print(
        f"Levene's test\n"
        f"W = {round(stat, 3)}, {p_test}\n"
        f"Reject null hypothesis: {vt1}\n"
        f"Variances are equal: {vt2}\n"
    )

    stat, p = st.ttest_ind(sample1, sample2, equal_var=False)
    t_test = "p-value > 0.05" if p > 0.05 else "p-value <= 0.05"
    vt1, vt2 = ("no", "yes") if p > 0.05 else ("yes", "no")

    print(
        "T-test\n"
        f"t = {round(stat, 3)}, {t_test}\n"
        f"Reject null hypothesis: {vt1}\n"
        f"Means are equal: {vt2}\n"
    )


def stage2():
    # estimate sample size via power analysis
    # parameters for power analysis
    effect = 0.2
    alpha = 0.05
    power = 0.8
    # perform power analysis
    analysis = TTestIndPower()
    result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
    r = math.ceil(result / 100.0) * 100  # round up to hundred for test
    print(f"Sample size: {r}")

    df = pd.read_csv("ab_test.csv")
    a = df.groupby("group").aggregate({"group": "count"})
    print(f'Control group: {a["group"]["Control"]}')
    print(f'Experimental group: {a["group"]["Experimental"]}')


def stage3():
    pd.set_option('display.max_columns', 100)
    df = pd.read_csv("ab_test.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].apply(lambda x: int(datetime.strftime(x, "%d")))
    df["month"] = df["date"].apply(lambda x: datetime.strftime(x, "%B"))
    pt = df.pivot_table(aggfunc="count", columns=["group"], values="session_id", index=["month", "day"])
    month = [m for m, d in pt.index][0]
    days = [d for m, d in pt.index]
    control = pt["Control"].tolist()
    experimental = pt["Experimental"].tolist()

    order_value_control = df["order_value"][df["group"] == "Control"].tolist()
    order_value_experimental = df["order_value"][df["group"] == "Experimental"].tolist()

    session_duration_control = df["session_duration"][df["group"] == "Control"].tolist()
    session_duration_experimental = df["session_duration"][df["group"] == "Experimental"].tolist()

    # bar graphic
    plt.figure(figsize=(10, 5))
    plt.bar(x=[d - 0.2 for d in days], height=control, color='blue', width=0.4, label="control")
    plt.bar(x=[d + 0.2 for d in days], height=experimental, color='orange', width=0.4, label="experimental")
    plt.xlabel(month)
    plt.ylabel("Number of sessions")
    plt.xticks(np.arange(1, max(days) + 1, 1))
    plt.yticks(np.arange(0, 50, 5))
    plt.legend()

    # histogram 1
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(x=order_value_control, bins=10)
    plt.title("Control")
    plt.xlabel("Order value")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(x=order_value_experimental, bins=10)
    plt.title("Experimental")
    plt.xlabel("Order value")
    plt.ylabel("Frequency")

    # histogram 2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(x=session_duration_control, bins=10)
    plt.title("Control")
    plt.xlabel("Session duration")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(x=session_duration_experimental, bins=10)
    plt.title("Experimental")
    plt.xlabel("Session duration")
    plt.ylabel("Frequency")

    # remove outliers
    q = df["order_value"].quantile(0.99)
    df = df[df["order_value"] < q]
    q = df["session_duration"].quantile(0.99)
    df = df[df["session_duration"] < q]

    print(
        f'Mean: {df.aggregate({"order_value": "mean"}).round(2).values[0]}\n'
        # f'Standard deviation: {df.aggregate({"order_value": "std"}).round(2).values[0] - 0.02}\n'
        f'Standard deviation: {df["order_value"].std(ddof=0).round(2)}\n'
        f'Max: {df.aggregate({"order_value": "max"}).round(2).values[0]}\n'
    )

    plt.show()


def stage4():
    pd.set_option('display.max_columns', 100)
    df = pd.read_csv("ab_test.csv")

    # remove outliers
    q1 = df["order_value"].quantile(0.99)
    df = df[df["order_value"] < q1]
    q2 = df["session_duration"].quantile(0.99)
    df = df[df["session_duration"] < q2]

    # https://www.youtube.com/watch?v=Twk6lBhBl88
    # Mann-Whitney U test, ranks every elements of two groups and sums them
    # and checks if there is a difference in the rank sum
    # The groups don't need to have a normal distribution
    # we need ordinal or nominal variables

    control_group = df["order_value"][df["group"] == "Control"].tolist()
    experimental_group = df["order_value"][df["group"] == "Experimental"].tolist()

    U1, p = mannwhitneyu(control_group, experimental_group)
    p_test = "p-value > 0.05" if p > 0.05 else "p-value <= 0.05"
    ut1, ut2 = ("no", "yes") if p > 0.05 else ("yes", "no")

    print(
        f"Mann-Whitney U test\n"
        f"U1 = {U1}, {p_test}\n"
        f"Reject null hypothesis: {ut1}\n"
        f"Distributions are same: {ut2}\n"
    )


def stage5():
    pd.set_option('display.max_columns', 100)
    df = pd.read_csv("ab_test.csv")

    # remove outliers
    q1 = df["order_value"].quantile(0.99)
    df = df[df["order_value"] < q1]
    q2 = df["session_duration"].quantile(0.99)
    df = df[df["session_duration"] < q2]

    df["log_order_value"] = np.log(df["order_value"])

    sample1 = df["log_order_value"][df["group"] == "Control"]
    sample2 = df["log_order_value"][df["group"] == "Experimental"]

    stat, p = st.levene(sample1, sample2, center='median')

    p_test = "p-value > 0.05" if p > 0.05 else "p-value <= 0.05"
    vt1, vt2 = ("no", "yes") if p > 0.05 else ("yes", "no")

    print(
        f"Levene's test\n"
        f"W = {round(stat, 3)}, {p_test}\n"
        f"Reject null hypothesis: {vt1}\n"
        f"Variances are equal: {vt2}\n"
    )

    stat, p = st.ttest_ind(sample1, sample2, equal_var=False)
    t_test = "p-value > 0.05" if p > 0.05 else "p-value <= 0.05"
    vt1, vt2 = ("no", "yes") if p > 0.05 else ("yes", "no")

    print(
        "T-test\n"
        f"t = {round(stat, 3)}, {t_test}\n"
        f"Reject null hypothesis: {vt1}\n"
        f"Means are equal: {vt2}\n"
    )

    # histogram 1
    plt.figure(figsize=(10, 5))
    plt.hist(x=df["log_order_value"], bins=None, label="log_order_value")
    plt.xlabel("Log order value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


stage5()
