import numpy as np
import time
import sys
sys.path.insert(0, "../DeepSequence/")
import model
import helper
import train

data_params = {
    #"dataset"           :   "BLAT_ECOLX"
    "dataset"           :   "rhla"
    }

model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "f_nonlin"          :  "sigmoid",
    "fps"               :   True,
    "n_pat"             :   4,
    "r_seed"            :   1,
    "conv_pat"          :   True,
    "d_c_size"          :   40,
    "sparsity_l"        :   1.0,
    "l2_l"              :   1.0,
    "dropout"           :   True,
    }

train_params = {
    "num_updates"       :   300000,
    "save_progress"     :   True,
    "verbose"           :   True,
    "save_parameters"   :   False,
    }

if __name__ == "__main__":

    data_helper = helper.DataHelper(dataset=data_params["dataset"],
                                    calc_weights=True)

    vae_model   = model.VariationalAutoencoderMLE(data_helper,
        batch_size                     =   model_params["bs"],
        encoder_architecture           =   [model_params["encode_dim_zero"],
                                                model_params["encode_dim_one"]],
        decoder_architecture           =   [model_params["decode_dim_zero"],
                                                model_params["decode_dim_one"]],
        n_latent                       =   model_params["n_latent"],
        logit_p                        =   model_params["logit_p"],
        encode_nonlinearity_type       =   "relu",
        decode_nonlinearity_type       =   "relu",
        final_decode_nonlinearity      =   model_params["f_nonlin"],
        final_pwm_scale                =   model_params["fps"],
        conv_decoder_size              =   model_params["d_c_size"],
        convolve_patterns              =   model_params["conv_pat"],
        n_patterns                     =   model_params["n_pat"],
        random_seed                    =   model_params["r_seed"],
        sparsity_lambda                =   model_params["sparsity_l"],
        l2_lambda                      =   model_params["l2_l"],
        sparsity                       =   model_params["sparsity"])

    job_string = helper.gen_job_string(data_params, model_params)

    print (job_string)

    train.train(data_helper, vae_model,
        num_updates             =   train_params["num_updates"],
        save_progress           =   train_params["save_progress"],
        save_parameters         =   train_params["save_parameters"],
        verbose                 =   train_params["verbose"],
        job_string              =   job_string)

    vae_model.save_parameters(file_prefix=job_string)
