import theano
theano.config.exception_verbosity = 'high'
import sys

new_path = '/thinker/glusterfs/home/TSLab-xcLu/DeepSequence-master/DeepSequence-master/DeepSequence'

if new_path not in sys.path:
    sys.path.append(new_path)

import model
import helper
import train
import csv

data_params = {"alignment_file":"CombGym/Baselines/DeepSequence/examples/alignments/SaCas9-KKH_b0.7.a2m"}

pabp_data_helper = helper.DataHelper(
                alignment_file=data_params["alignment_file"],
                working_dir="CombGym/Baselines/DeepSequence",
                calc_weights=False
                )

model_params = {
        "batch_size"        :   100,
        "encode_dim_zero"   :   1500,
        "encode_dim_one"    :   1500,
        "decode_dim_zero"   :   100,
        "decode_dim_one"    :   500,
        "n_patterns"        :   4,
        "n_latent"          :   30,
        "logit_p"           :   0.001,
        "sparsity"          :   "logit",
        "encode_nonlin"     :   "relu",
        "decode_nonlin"     :   "relu",
        "final_decode_nonlin":  "sigmoid",
        "output_bias"       :   True,
        "final_pwm_scale"   :   True,
        "conv_pat"          :   True,
        "d_c_size"          :   40
        }

pabp_vae_model   = model.VariationalAutoencoder(pabp_data_helper,
    batch_size              =   model_params["batch_size"],
    encoder_architecture    =   [model_params["encode_dim_zero"],
                                model_params["encode_dim_one"]],
    decoder_architecture    =   [model_params["decode_dim_zero"],
                                model_params["decode_dim_one"]],
    n_latent                =   model_params["n_latent"],
    n_patterns              =   model_params["n_patterns"],
    convolve_patterns       =   model_params["conv_pat"],
    conv_decoder_size       =   model_params["d_c_size"],
    logit_p                 =   model_params["logit_p"],
    sparsity                =   model_params["sparsity"],
    encode_nonlinearity_type       =   model_params["encode_nonlin"],
    decode_nonlinearity_type       =   model_params["decode_nonlin"],
    final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
    output_bias             =   model_params["output_bias"],
    final_pwm_scale         =   model_params["final_pwm_scale"],
    working_dir             =   ".")

print ("Model built")

file_prefix = "SaCas9-KKH_b0.7"
pabp_vae_model.load_parameters(file_prefix=file_prefix)
print ("Parameters loaded")

pabp_custom_matr_mutant_name_list, pabp_custom_matr_delta_elbos \
    = pabp_data_helper.custom_mutant_matrix("data/SaCas9_mean.csv", \
                                            pabp_vae_model, N_pred_iterations=500)
with open('/thinker/glusterfs/home/TSLab-xcLu/DeepSequence-master/DeepSequence-master/examples/results/SaCas9-KKH_b0.7.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Mutant Name", "Delta Elbos"])  
    writer.writerows(zip(pabp_custom_matr_mutant_name_list, pabp_custom_matr_delta_elbos))  

    
print (pabp_custom_matr_mutant_name_list, pabp_custom_matr_delta_elbos)

