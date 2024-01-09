# Drug-Discovery-Using-ML

# 1st Module
import pandas as pd

import numpy as np

#ChEMBL webresource client : The library helps accessing ChEMBL data and cheminformatics tools from Python.

from chembl_webresource_client.new_client import new_client

# target search for Dengue virus

target = new_client.target

print(target)

![image](https://github.com/vinaya1699/Drug-Discovery-Using-ML/assets/110582335/1e845dd0-0553-425d-af3f-15835ac9dc90)

target_query = target.search("Dengue virus")

print(target_query)

![image](https://github.com/vinaya1699/Drug-Discovery-Using-ML/assets/110582335/c8926ef8-8383-4b1d-918e-00f08112c3de)



target_df = pd.DataFrame.from_dict(target_query)

print(target_df)

![image](https://github.com/vinaya1699/Drug-Discovery-Using-ML/assets/110582335/a6c92aa3-4b50-4f4e-b297-2833073265f7)


target_df[(target_df['target_type']=='SINGLE PROTEIN') & (target_df['organism']=='Dengue virus')]

selected_target = target_df.target_chembl_id[6]
print(selected_target)

activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type='IC50')

df = pd.DataFrame.from_dict(res)
print(df.head())

df.standard_type.unique()

df.to_csv('bioactivity_data_dengue.csv', index=False)

df_clean = df[df['standard_value'].notna()]
df_clean = df_clean[df_clean['canonical_smiles'].notna()]
df_clean.drop_duplicates(['canonical_smiles'], inplace = True)

print(df_clean.shape)
df_clean.head()

selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df_activity = df_clean[selection]

print(df_activity.head())

def activity_classifier(sv):

    if float(sv) >= 10000:
        return "inactive"
    elif float(sv) <= 1000:
        return "active"
    else:
        return "intermediate"

df_activity['class'] = df_activity['standard_value'].apply(activity_classifier)

print(df_activity.head())

df_activity.to_csv("bioactivity_dengue_preprocessed_data.csv", index=False)

###################################################################################################################################

# 2nd Module

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# calculate descriptors and lipinski values

df_proc = pd.read_csv("bioactivity_dengue_preprocessed_data.csv")
print(df_proc)

df_proc['canonical_smiles_mol'] = df_proc['canonical_smiles'].apply(Chem.MolFromSmiles)
df_proc['mol_wt'] = df_proc['canonical_smiles_mol'].apply(Descriptors.MolWt)
df_proc['mol_logp'] = df_proc['canonical_smiles_mol'].apply(Descriptors.MolLogP)
df_proc['num_H_donors'] = df_proc['canonical_smiles_mol'].apply(Lipinski.NumHDonors)
df_proc['num_H_acceptors'] = df_proc['canonical_smiles_mol'].apply(Lipinski.NumHAcceptors)

print(df_proc.head())

def norm(x):
    """
    to limit the standard value to 100  million
    """
    if x > 100000000:
        x = 100000000
    return x

restricting the standard value by applying the norm function and log10

df_proc['pIC50'] = np.log10(df_proc['standard_value'].apply(norm))
print(df_proc.head())

getting statistics of pIC50 values

df_proc['pIC50'].describe()

dropping the intermediate class

df2 = df_proc[df_proc['class'] != 'intermediate']
print(df2.head())

######################################################################################################################################

# 3rd Module

import seaborn as sns
sns.set(style= 'ticks')
import matplotlib.pyplot as plt
%matplotlib inline

# frequency of Bioactivity class

sns.countplot(x='class', data=df2)

plt.xlabel('Bioactivity class')
plt.ylabel('Frequency')

plt.savefig('plot_bioactivity_class.pdf')

plt.show()

molecular weight and log of standard P value

sns.scatterplot(x='mol_wt', y='mol_logp', data=df2, hue='class', size='pIC50', alpha = 0.7)

plt.xlabel('Molecular Weght')
plt.ylabel('LogP')
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)
plt.savefig('plot_mw_vs_logp.pdf')

plt.show()

###################################################################################################################################

# Running Mann-Whitney U test for statstically significant difference between classes

from numpy.random import seed, randn
from scipy.stats import mannwhitneyu

seed (1)

def whhitney_test(descriptor, verbose=False):

    """
    function to run mann-whitney U test
    """

    selection = [descriptor, 'class']
    df = df2[selection]
    active = df[df['class'] == 'active']
    active = active[descriptor]

    selection = [descriptor, 'class']
    df = df2[selection]
    inactive = df[df['class'] == 'inactive']
    inactive = inactive[descriptor]

    compare samples
    stat, p = mannwhitneyu(active, inactive)

    interpretattion
    alpha = 0.05
    if p > alpha:
        inter = "No significant difference"
    else:
        inter = "Distribution significantly different"

    results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':inter}, index=[0])
    filename = 'mannwhitneyu_' + descriptor + '.csv'
    results.to_csv(filename)

    return results

# bioactivity class and pIC50 value

sns.boxplot(x='class', y='pIC50', data=df2)
plt.xlabel('Bioactivity class')
plt.ylabel('pIC50 Value')

plt.savefig('plot_bioactivity_class.pdf')

plt.show()

# whitney test for pIC50
whhitney_test('pIC50')

# bioactivity class and molecular weight value

sns.boxplot(x='class', y='mol_wt', data=df2)
plt.xlabel('Bioactivity class')
plt.ylabel('Molecular Weight')

plt.savefig('plot_bioactivity_class.pdf')

plt.show()

# whitney test for pIC50
whhitney_test('mol_wt')

# bioactivity class and log of P value

sns.boxplot(x='class', y='mol_logp', data=df2)
plt.xlabel('Bioactivity class')
plt.ylabel('logP')

plt.savefig('plot_bioactivity_class_mol_logp.pdf')

plt.show()

# whitney test for mol_logP
whhitney_test('mol_logp')

# bioactivity class and num_H_donors

sns.boxplot(x='class', y='num_H_donors', data=df2)
plt.xlabel('Bioactivity class')
plt.ylabel('Number of H donors')

plt.savefig('plot_bioactivity_class.pdf')

plt.show()

# whitney test for num_H_donors
whhitney_test('num_H_donors')

# bioactivity class and num_H_acceptors

sns.boxplot(x='class', y='num_H_acceptors', data=df2)
plt.xlabel('Bioactivity class')
plt.ylabel('Number of H acceptors')

plt.savefig('plot_bioactivity_class.pdf')

plt.show()

# whitney test for number of H acceptors
whhitney_test('num_H_acceptors')

######################################################################################################################################

# 4rth Module

df3 = df_proc.copy()

print(df3.head())

selection = ['canonical_smiles', 'molecule_chembl_id']
df3_selection = df3[selection]
df3_selection.to_csv('molecule.smi', sep='\t', index = False, header = False)

print(df3_selection.head())

from padelpy import padeldescriptor

padeldescriptor(mol_dir='molecule.smi', d_file='descriptors.csv', 
                fingerprints=True, removesalt=True, detectaromaticity=True, 
                log=True, retainorder=True)

df3_padel = pd.read_csv('descriptors.csv')
print(df3_padel)

df3_padel = df3_padel.drop(columns='Name')
print(df3_padel.head())

X_1 = df3_padel.copy()
Y_1 = df3['pIC50']
data_xy = pd.concat([X_1, Y_1], axis=1)
print(data_xy.head())

data_xy.to_csv('data with descriptors padel.csv', index=False)
X = data_xy.drop('pIC50', axis = 1)
Y = data_xy['pIC50']
X.shape
Y.shape

######################################################################################################################################

# 5th Module

from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(0.8*(1-0.8)))
X = selection.fit_transform(X)
X.shape

# data split into train and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

x_train.shape, y_train.shape

from lazypredict.Supervised import LazyRegressor

clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
train, test = clf.fit(x_train, x_test, y_train, y_test)

print(train)

print(test)

from sklearn.linear_model import SGDRegressor

np.random.seed(42)

sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(x_train, y_train)

s1 = sgd_model.score(x_test, y_test)
print(s1)

y_pred = sgd_model.predict(x_test)

# plotting predicted vs actual data

ax = sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.4})
ax.set_xlabel('Experimental pIC50')
ax.set_ylabel('Predicted pIC50')
ax.set_xlim(0,6)
ax.set_ylim(0,6)

plt.show()
