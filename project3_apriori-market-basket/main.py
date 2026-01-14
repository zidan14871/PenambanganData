import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Sample dataset transaksi
dataset = [
    ['Bread', 'Milk'],
    ['Bread', 'Diapers', 'Beer', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Coke'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Coke']
]

# One-hot encode
te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

# Frequent itemsets
freq_items = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

print("=== Apriori Market Basket Analysis ===")
print("\nFrequent Itemsets:\n", freq_items)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
