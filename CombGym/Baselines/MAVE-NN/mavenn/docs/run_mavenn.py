import mavenn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import ndcg_score
import math
from scipy import stats


pd.options.mode.chained_assignment = None  # default='warn'


protein_name = 'SaCas9'
model_name = 'mavenn'
data_folder = 'MAVE-NN\mavenn\docs\datasets\SaCas9\splits'
model_save_folder = '/home/Lxc/mavenn/mavenn/examples/baseline_parameters/'
results_save_folder = '/home/Lxc/mavenn/mavenn/examples/baseline_results/SaCas9/'


csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and '0vsrest' not in f]


for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    
    # Load dataset
    data_df = pd.read_csv(os.path.join(data_folder, csv_file))

    # Get and report sequence length
    L = len(data_df.loc[0, 'sequence'])
    print(f"Sequence length: {L:d} amino acids (+ stops)")
    
    # Preview dataset
    data_df.loc[data_df['validation'] == True, 'set'] = 'validation'
    data_df['set'] = data_df['set'].replace('train', 'training')
    data_df = data_df.drop(columns=['validation'])

    # Split dataset
    trainval_df, test_df = mavenn.split_dataset(data_df)

    # Define model
    model = mavenn.Model(L=L, alphabet='protein*', gpmap_type='blackbox', regression_type='GE', 
                         ge_noise_model_type='SkewedT', ge_heteroskedasticity_order=2)
    if not (data_df[data_df['set'] == 'training']['n_mut'] > 1).sum():
        model.ge_nonlinearity_type = 'linear'
        model.gpmap_type = 'additive'
        print('only single mutant')

    #print(data_df.head())

    # Set training data
    model.set_data(x=trainval_df['sequence'], y=trainval_df['target'], validation_flags=trainval_df['validation'])

    # Train model
    model.fit(learning_rate=1e-3, epochs=500, batch_size=64, early_stopping=True, early_stopping_patience=25, verbose=False)

    # Compute variational information
    I_var, dI_var = model.I_variational(x=test_df['sequence'], y=test_df['target'])
    print(f'test_I_var: {I_var:.3f} +- {dI_var:.3f} bits')

    I_var, dI_var = model.I_variational(x=trainval_df['sequence'], y=trainval_df['target'])
    print(f'trainval_I_var: {I_var:.3f} +- {dI_var:.3f} bits')

    # Compute predictive information on trainval data
    I_pred, dI_pred = model.I_predictive(x=test_df['sequence'], y=test_df['target'])
    print(f'test_I_pred: {I_pred:.3f} +- {dI_pred:.3f} bits')

    I_pred, dI_pred = model.I_predictive(x=trainval_df['sequence'], y=trainval_df['target'])
    print(f'trainval_I_pred: {I_pred:.3f} +- {dI_pred:.3f} bits')

    # Save model to file
    #model_file_name = f"{protein_name}_{model_name}_{csv_file.split('.')[0]}"
    model_file_name = "test"
    model_path = os.path.join(model_save_folder, model_file_name)
    model.save(model_path)

    # Get test data y values
    y_test = test_df['target']

    # Compute yhat on test data
    yhat_test = model.x_to_yhat(test_df['sequence'])

    # Assemble into dataframe
    final_df = pd.DataFrame({'yhat_test': yhat_test, 'y_test': y_test})

    results_file_name = f"{protein_name}_{model_name}_{csv_file.split('.')[0]}.csv"
    results_path = os.path.join(results_save_folder, results_file_name)
    final_df.to_csv(results_path, index=True)

    print(f"Model and results saved for file: {csv_file}\n")
