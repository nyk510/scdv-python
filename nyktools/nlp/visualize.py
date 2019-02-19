# coding: utf-8
"""
可視化ツール
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_topics(ldamodel, show_all=True, num_token=10, figsize=(6, 6)):
    """
    ldamodelのトピック中で高頻度に現れるtokenをbarplotにより可視化

    Args:
        ldamodel (gensim.models.LdaModel)
        show_all (True or int)
    """
    if show_all:
        n_topics = ldamodel.num_topics
    else:
        n_topics = show_all

    for topic_id in range(n_topics):
        data_i = np.array(ldamodel.show_topic(topic_id, topn=num_token))
        labels = data_i[:, 0]
        probs = data_i[:, 1].astype(np.float)
        df_i = pd.DataFrame(probs, index=labels, columns=["probability"])

        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        df_i.plot(kind="barh", ax=ax1)
        yield fig, ax1
