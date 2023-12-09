import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import auc

def analyse_results(df, name):
    df['achieves_sufficient_explanation'] = df['original_prediction'] * df['prediction_explanation_events_only'] > 0
    
    necessary_explanations = df[df['achieves_counterfactual_explanation'] == True]
    sufficient_explanations = df[df['achieves_sufficient_explanation'] == True]
    necessary_explanation_rate = len(necessary_explanations)/len(df)
    sufficient_explanation_rate = len(sufficient_explanations)/len(df)
    sparsity_values_fid_plus = []
    sparsity_necessary = 0.0
    for necessary_example_event_ids, candidate_size in zip(necessary_explanations['cf_example_event_ids'].to_numpy(), necessary_explanations['candidate_size'].to_numpy()):
        sparsity_values_fid_plus.append((len(necessary_example_event_ids)/candidate_size))
        sparsity_necessary += (len(necessary_example_event_ids)/candidate_size)
    if len(necessary_explanations) > 0:
        sparsity_necessary = sparsity_necessary / len(necessary_explanations)
    else:
        sparsity_necessary = 0
    
    duration = df['total_duration'].mean() / 1000000000
    init_duration = df['init_duration'].mean() / 1000000000
    oracle_call_duration = df['oracle_call_duration'].mean() / 1000000000
    explanation_duration = df['explanation_duration'].mean() / 1000000000

    if len(sparsity_values_fid_plus) > 0:
        sparsity_list_fid_plus = [np.min(sparsity_values_fid_plus) - 0.00001]
    else:
        sparsity_list_fid_plus = [0.0]
    necessary_rate_list = [0.0]
    found_explanations = 0
    found_sparsities = []
    for value in sorted(set(sparsity_values_fid_plus), reverse=False):
        num_examples_with_sparsity = len([val for val in sparsity_values_fid_plus if val == value])
        found_explanations += num_examples_with_sparsity
        necessary_rate_list.append(found_explanations/200.0)
        found_sparsities += [value] * num_examples_with_sparsity
        sparsity_list_fid_plus.append(value)
    sparsity_list_fid_plus.append(1.0)
    necessary_rate_list.append(necessary_rate_list[-1])

    auc_fid_plus = auc(sparsity_list_fid_plus, necessary_rate_list)

    



    sparsity_values_fid_min = []
    sparsity_sufficient = 0.0
    for sufficient_example_event_ids, candidate_size in zip(sufficient_explanations['cf_example_event_ids'].to_numpy(), sufficient_explanations['candidate_size'].to_numpy()):
        sparsity_values_fid_min.append((len(sufficient_example_event_ids)/candidate_size))
        sparsity_sufficient += (len(sufficient_example_event_ids)/candidate_size)
    if len(sufficient_explanations) > 0:
        sparsity_sufficient = sparsity_sufficient / len(sufficient_explanations)
    else:
        sparsity_sufficient = 0
    

    if len(sparsity_values_fid_min) > 0:
        sparsity_list_fid_min = [np.min(sparsity_values_fid_min) - 0.00001]
    else:
        sparsity_list_fid_min = [0.0]
    sufficient_rate_list = [0.0]
    found_explanations = 0
    found_sparsities = []
    for value in sorted(set(sparsity_values_fid_min), reverse=False):
        num_examples_with_sparsity = len([val for val in sparsity_values_fid_min if val == value])
        found_explanations += num_examples_with_sparsity
        sufficient_rate_list.append(found_explanations/200.0)
        found_sparsities += [value] * num_examples_with_sparsity
        sparsity_list_fid_min.append(value)
    sparsity_list_fid_min.append(1.0)
    sufficient_rate_list.append(sufficient_rate_list[-1])

    auc_fid_min = auc(sparsity_list_fid_min, sufficient_rate_list)
    
    fidelity_min = df['achieves_sufficient_explanation'].sum() / len(df)
    
    fidelity_plus = df['achieves_counterfactual_explanation'].sum() / len(df)
    
    return {
        'Selection strategy': name,
        'necessary explanation rate': necessary_explanation_rate,
        'sufficient explanation rate': sufficient_explanation_rate,
        'sparsity_necessary': sparsity_necessary,
        'sparsity_sufficient': sparsity_sufficient,
        'avg_oracle_calls': df['oracle_calls'].sum() / len(df),
        'avg_oracle_calls_cf': necessary_explanations['oracle_calls'].sum() / len(necessary_explanations),
        'Duration': duration,
        'initialisation (s)': init_duration,
        'oracle calls (s)': oracle_call_duration,
        'explanation (s)': explanation_duration,
        'sparsity_values_fid_plus': sparsity_values_fid_plus,
        'sparsity_list_fid_plus': sparsity_list_fid_plus,
        'sparsity_values_fid_min': sparsity_values_fid_min,
        'sparsity_list_fid_min': sparsity_list_fid_min,
        'AUFC_plus': auc_fid_plus,
        'AUFC_min': auc_fid_min,
        'necessary_rate_list': necessary_rate_list,
        'sufficient_rate_list': sufficient_rate_list,
        'fidelity_minus': fidelity_min,
        'fidelity_plus': fidelity_plus,
        'sparsity_all': (df['cf_example_event_ids'].apply(lambda x: len(x)) / df['candidate_size']).mean(),
        'characterization_score': 1/((0.5/fidelity_plus)+(0.5/fidelity_min))
    }


def expand_sparsity_explanation_rate_necessary(df: pd.DataFrame):
    data = []
    for index, row in df.iterrows():
        for sparsity_necessary, explanation_rate in zip(row['sparsity_list_fid_plus'], row['necessary_rate_list']):
            data.append({
                'sparsity_necessary': sparsity_necessary,
                'necessary explanation rate': explanation_rate,
                'Explainer': row['Explainer'],
                'Selection strategy': row['Selection strategy']
            })
    return pd.DataFrame(data)

def expand_sparsity_explanation_rate_sufficient(df: pd.DataFrame):
    data = []
    for index, row in df.iterrows():
        for sparsity_sufficient, explanation_rate in zip(row['sparsity_list_fid_min'], row['sufficient_rate_list']):
            data.append({
                'sparsity_sufficient': sparsity_sufficient,
                'sufficient explanation rate': explanation_rate,
                'Explainer': row['Explainer'],
                'Selection strategy': row['Selection strategy']
            })
    return pd.DataFrame(data)


def calculate_similarity_scores(results, results_other, necessary_explanations_only:bool = False):
    jaccard_similarities = []
    precisions = []
    recalls = []
    f1s = []
    subset_accuracies = 0

    if necessary_explanations_only:
        results = results[np.isin(results['explained_event_id'], results_other['explained_event_id'])].reset_index(drop=True)
        results_other = results_other[np.isin(results_other['explained_event_id'], results['explained_event_id'])].reset_index(drop=True)
        results = results[results['achieves_counterfactual_explanation'] | results_other['achieves_counterfactual_explanation']].reset_index(drop=True)
    
    for index, row in results.iterrows():
        try:
            other_row = results_other[results_other['explained_event_id'] == row['explained_event_id']].iloc[0]

            a = row['cf_example_event_ids']
            b = other_row['cf_example_event_ids']
            true_positives = np.sum(np.isin(a,b))
            false_positives = np.sum(~np.isin(a, b))
            false_negatives = np.sum(~np.isin(b, a))

            if false_positives == 0:
                subset_accuracies += 1
            
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
    return np.average(precisions), np.average(recalls), np.average(f1s), np.average(jaccard_similarities), subset_accuracies / len(results)


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
