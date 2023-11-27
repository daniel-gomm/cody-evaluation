import pandas as pd
import numpy as np

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
        'sparsity_all': (df['cf_example_event_ids'].apply(lambda x: len(x)) / df['candidate_size']).mean()
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