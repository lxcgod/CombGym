import re
from model import CouplingsModel
import pandas as pd
import csv

c = CouplingsModel("parameters/HIV-1_b0.5.model_params")
datas = []
with open('data/HIV-1.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    
    for row in reader:
        parsed_mutations = []
        mutations = re.split(r'[;:-]', row[0])  
        mutations = re.findall(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", row[0])
        for mutation in mutations:
            original_amino_acid = mutation[0]
            position = int(mutation[1])
            mutated_amino_acid = mutation[2]
            parsed_mutations.append((position, original_amino_acid, mutated_amino_acid))
        
        print(parsed_mutations)   
        try: 
            delta_E, delta_E_couplings, delta_E_fields = c.delta_hamiltonian(parsed_mutations)
            print(delta_E, delta_E_couplings, delta_E_fields)
            datas.append([parsed_mutations,delta_E])
        except Exception as e:
            datas.append([parsed_mutations,"failed"])    
df = pd.DataFrame(datas,columns=['orignal_data','delta_E'])
df.to_csv('result/HIV-1_b0.5.csv', index=False)