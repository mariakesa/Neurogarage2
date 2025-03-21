import pandas as pd

dat=pd.read_csv('/home/maria/Neurogarage2/scotusCebra/scotus_with_summaries_ordered.csv')
print(dat.loc[3]['summary'])