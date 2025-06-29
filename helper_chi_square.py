import itertools
import pandas as pd
from scipy.stats import chi2_contingency

def chi_square_all_pairs(df, categorical_cols, alpha=0.05):
    results = []
    for col1, col2 in itertools.permutations(categorical_cols, 2):
        contingency = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, _ = chi2_contingency(contingency)
        result = {
            'Var1': col1,
            'Var2': col2,
            'Chi2': round(chi2, 4),
            'p-value': round(p, 4),
            'dof': dof,
            'Independent?': 1 if p > alpha else 0
        }
        results.append(result)
    return pd.DataFrame(results)