import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules

## What are the product combinations purchased together by customers at Allias Megastore?
# Import data
file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D599/Task3-D599/Megastore Dataset.csv"
data = pd.read_csv(file_path)

# Part 3

# Encoding Methods
# One-Hot Encoding for 'PaymentMethod'
df_encoded = pd.get_dummies(data, columns=['PaymentMethod'], drop_first=True)

# Leave 'Order ID' as is 

# Ordinal Encoding for 'CustomerOrderSatisfaction'
satisfaction_mapping = {
    'Very Satisfied': 4,
    'Satisfied': 3,
    'Dissatisfied': 1,
    'Very Dissatisfied': 2,
    'Prefer to not respond': 0
}
df_encoded['CustomerOrderSatisfaction'] = df_encoded['CustomerOrderSatisfaction'].map(satisfaction_mapping)

# Ordinal Encoding for 'OrderPriority'
priority_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3
}
df_encoded['OrderPriority'] = df_encoded['OrderPriority'].map(priority_mapping)

# Display only the encoded variables
encoded_columns = ['OrderID', 'CustomerOrderSatisfaction', 'OrderPriority'] + [col for col in df_encoded.columns if 'PaymentMethod_' in col]
df_encoded_only = df_encoded[encoded_columns]
print(df_encoded_only)

# Export DataFrame to a new CSV file
file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D599/Task3-D599/Task3-D599_Encoded.csv"
df_encoded_only.to_csv(file_path, index=False)

print(f"Encoded file saved at: {file_path}")

# Transactionalized Data
# Create a transactionalized DataFrame
transactional_df = data.groupby('OrderID')['ProductName'].apply(lambda x: ', '.join(x)).reset_index()

# Rename the columns for clarity
transactional_df.columns = ['OrderID', 'ProductCombinations']

# Display the transactional DataFrame
print(transactional_df)

# Export the transactional DataFrame to a new CSV file
transactional_file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D599/Task3-D599/Task3-D599_Transactional.csv"
transactional_df.to_csv(transactional_file_path, index=False)

print(f"Transactional DataFrame saved at: {transactional_file_path}")

# Association rules with the Apriori algorithm
# Split the 'ProductCombinations' into lists
transactional_df['ProductCombinations'] = transactional_df['ProductCombinations'].apply(lambda x: x.split(', '))

# Use MultiLabelBinarizer to create a one-hot encoded matrix
mlb = MultiLabelBinarizer()
transaction_matrix = mlb.fit_transform(transactional_df['ProductCombinations'])

# Convert to a DataFrame
transaction_df = pd.DataFrame(transaction_matrix, columns=mlb.classes_, index=transactional_df['OrderID'])

# Apply the Apriori algorithm
frequent_itemsets = apriori(transaction_df, min_support=0.05, use_colnames=True)

# Generate the association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the rules
print(rules)

# Export the rules to a new CSV file
rules_file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D599/Task3-D599/Task3-D599_Association_Rules.csv"
rules.to_csv(rules_file_path, index=False)

print(f"Association rules saved at: {rules_file_path}")

# Support, Lift and Confidence Values:
filtered_rules = rules[['support', 'confidence', 'lift']]
print(filtered_rules)

