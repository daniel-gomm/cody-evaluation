import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def analyse_results(df, name):
    cf_explanations = df[df['achieves_counterfactual_explanation'] == True]
    cf_explanation_rate = len(cf_explanations)/len(df)
    sparsity_values = []
    sparsity = 0.0
    for cf_example_event_ids, candidate_size in zip(cf_explanations['cf_example_event_ids'].to_numpy(), cf_explanations['candidate_size'].to_numpy()):
        sparsity_values.append((len(cf_example_event_ids)/candidate_size))
        sparsity += (len(cf_example_event_ids)/candidate_size)
    if len(cf_explanations) > 0:
        sparsity = sparsity / len(cf_explanations)
    else:
        sparsity = 0
    
    duration = df['total_duration'].mean() / 1000000000
    init_duration = df['init_duration'].mean() / 1000000000
    oracle_call_duration = df['oracle_call_duration'].mean() / 1000000000
    explanation_duration = df['explanation_duration'].mean() / 1000000000

    if len(sparsity_values) > 0:
        sparsity_list = [np.min(sparsity_values) - 0.00001]
    else:
        sparsity_list = [0.0]
    explanation_rate_list = [0.0]
    found_explanations = 0
    found_sparsities = []
    for value in sorted(set(sparsity_values), reverse=False):
        num_examples_with_sparsity = len([val for val in sparsity_values if val == value])
        found_explanations += num_examples_with_sparsity
        explanation_rate_list.append(found_explanations/200.0)
        found_sparsities += [value] * num_examples_with_sparsity
        sparsity_list.append(value)
    sparsity_list.append(1.0)
    explanation_rate_list.append(explanation_rate_list[-1])
    
    df['achieves_sufficient_explanation'] = df['original_prediction'] * df['prediction_explanation_events_only'] > 0
    
    fidelity_plus = df['achieves_sufficient_explanation'].sum() / len(df)
    
    fidelity_minus = df['achieves_counterfactual_explanation'].sum() / len(df)
    
    return {
        'Selection strategy': name,
        'explanation rate': cf_explanation_rate,
        'sparsity': sparsity,
        'avg_oracle_calls': df['oracle_calls'].sum() / len(df),
        'avg_oracle_calls_cf': cf_explanations['oracle_calls'].sum() / len(cf_explanations),
        'Duration': duration,
        'initialisation (s)': init_duration,
        'oracle calls (s)': oracle_call_duration,
        'explanation (s)': explanation_duration,
        'sparsity_values': sparsity_values,
        'sparsity_list': sparsity_list,
        'explanation_rate_list': explanation_rate_list,
        'fidelity_plus': fidelity_plus,
        'fidelity_minus': fidelity_minus,
        'sparsity_all': (df['cf_example_event_ids'].apply(lambda x: len(x)) / df['candidate_size']).mean(),
        'characterization_score': 1/((0.5/fidelity_plus)+(0.5/fidelity_minus))
    }


def expand_sparsity_explanation_rate(df: pd.DataFrame):
    data = []
    for index, row in df.iterrows():
        for sparsity, explanation_rate in zip(row['sparsity_list'], row['explanation_rate_list']):
            data.append({
                'sparsity': sparsity,
                'explanation rate': explanation_rate,
                'Explainer': row['Explainer'],
                'Selection strategy': row['Selection strategy']
            })
    return pd.DataFrame(data)


def calculate_similarity_scores(results, results_other):
    jaccard_similarities = []
    precisions = []
    recalls = []
    f1s = []
    aymaras = 0
    for index, row in results.iterrows():
        try:
            other_row = results_other[results_other['explained_event_id'] == row['explained_event_id']].iloc[0]

            a = row['cf_example_event_ids']
            b = other_row['cf_example_event_ids']
            true_positives = np.sum(np.isin(a,b))
            false_positives = np.sum(~np.isin(a, b))
            false_negatives = np.sum(~np.isin(b, a))

            if false_positives == 0:
                aymaras += 1
            
            epsilon = 1e-9 # Use epsilon as a smoothing factor to mitigate cases where we would otherwise divide by zero

            precision = (true_positives + epsilon) / (true_positives + false_positives + epsilon)
            precisions.append(precision)
            
            recall = (true_positives+epsilon) / (true_positives + false_negatives + epsilon)
            recalls.append(recall)
            
            f1 = (2 * precision * recall) / (precision + recall)
            f1s.append(f1)
            
            jaccard_similarity = len(np.intersect1d(a, b)) / len(np.union1d(a, b))
            jaccard_similarities.append(jaccard_similarity)
        except:
            pass
    return np.average(precisions), np.average(recalls), np.average(f1s), np.average(jaccard_similarities), aymaras / len(results)


# From https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
