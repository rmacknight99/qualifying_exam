import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def reduce_columns(df):
    cols = list(df.columns)
    df = df[df.SN1 != 'check']
    df['Nucleophile'] = [int(string[4]) for string in df['Nucleophile']]
    amounts = []
    LAs = []
    for string in df['Lewis Acid']:
        if string == '-':
            LA = 'none'
            amount = 0.0
        for i, character in enumerate(string):
            if character == ':':
               if string[i+6] == 'e':
                   amount = float(string[i+2:i+5])
               else:
                   amount = float('1.0')
               LA = string[0:i]
        LAs.append(LA)
        amounts.append(amount)
    df['Conversion'] = [round(float(i)/100,2) for i in df['Conversion']]
    df['SN2'] = [round(float(i)/100,2) for i in df['SN2']]
    df['SN1'] = [round(float(i)/100,2) for i in df['SN1']]
    
    if 'Time' in cols:
        time = [int(string[0]) for string in df['Time']]
        temp = df['Temperature']

    reduced_df = pd.DataFrame()
    reduced_df['nucleophile_eq'] = df['Nucleophile']
    try:
        reduced_df['time'] = time
        print("found time and temp...writing")
        reduced_df['temp'] = temp
    except:
        print("no time and temp found...assuming constant")
    reduced_df['lewis_acid_eq'] = amounts
    reduced_df['lewis_acid'] = LAs
    reduced_df['solvent'] = df['Solvent']
    reduced_df['conversion'] = df['Conversion']
    reduced_df['sn2'] = df['SN2']
    reduced_df['sn1'] = df['SN1']

    reduced_df = reduced_df[reduced_df["lewis_acid"] != "none"]
    reduced_df = reduced_df[reduced_df["solvent"] == "ACN"]
    
    feature_importance_df = reduced_df

    reduced_df = reduced_df[reduced_df["nucleophile_eq"] == 2]
    reduced_df = reduced_df[reduced_df["lewis_acid_eq"] == 0.1]
    
    return reduced_df, feature_importance_df


