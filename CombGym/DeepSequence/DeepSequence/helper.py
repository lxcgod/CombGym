from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from collections import defaultdict
import _pickle
import os


class DataHelper:
    def __init__(self,
        dataset="",
        alignment_file="",
        focus_seq_name="",
        calc_weights=True,
        working_dir="",
        theta=0.2,
        load_all_sequences=True,
        alphabet_type="protein"):

        """
        Class to load and organize alignment data.
        This function also helps makes predictions about mutations.

        Parameters
        --------------
        dataset: preloaded dataset names
                    We have found it easiest to organize datasets in this
                    way and use the self.configure_datasets() func
        alignment_file: Name of the alignment file located in the "datasets"
                            folder. Not needed if dataset pre-entered
        focus_seq_name: Name of the sequence in the alignment
                            Defaults to the first sequence in the alignment
        calc_weights: (bool) Calculate sequence weights
                        Default True, but not necessary if just loading weights
                            and doing mutation effect prediction
        working_dir: location of "params", "logs", "embeddings", and "datasets"
                        folders
        theta: Sequence weighting hyperparameter
                Generally: Prokaryotic and eukaryotic families =  0.2
                            Viruses = 0.01
        load_all_sequences:
        alphabet_type: Alphabet type of associated dataset.
                            Options are DNA, RNA, protein, allelic

        Returns
        ------------
        None
        """

        np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type

        # Initalize the elbo of the wt to None
        #   will be useful if eventually doing mutation effect prediction
        self.wt_elbo = None

        # Alignment processing parameters
        self.theta = theta

        # If I am running tests with the model, I don't need all the
        #    sequences loaded
        self.load_all_sequences = load_all_sequences

        # Load necessary information for preloaded datasets
        if self.dataset != "":
            self.configure_datasets()

        # Load up the alphabet type to use, whether that be DNA, RNA, or protein
        if self.alphabet_type == "protein":
            self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
            self.reorder_alphabet = "DEKRHNQSTPGAVILMCFYW"
        elif self.alphabet_type == "RNA":
            self.alphabet = "ACGU"
            self.reorder_alphabet = "ACGU"
        elif self.alphabet_type == "DNA":
            self.alphabet = "ACGT"
            self.reorder_alphabet = "ACGT"
        elif self.alphabet_type == "allelic":
            self.alphabet = "012"
            self.reorder_alphabet = "012"

        #then generate the experimental data
        self.gen_basic_alignment()

        if self.load_all_sequences:
            self.gen_full_alignment()

    def configure_datasets(self):

        if self.dataset == "BLAT_ECOLX":
            self.alignment_file = self.working_dir+"/datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"
            self.theta = 0.2
        
        elif self.dataset == "SaCas9-KKH_b0.7":
            self.alignment_file = "CombGym/Baselines/DeepSequence/examples/alignments/SaCas9-KKH_b0.7.a2m"
            self.theta = 0.2


        elif self.dataset == "DLG4_RAT":
            self.alignment_file = self.working_dir+"/datasets/DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50.a2m"
            self.theta = 0.2

        elif self.dataset == "trna":
            self.alignment_file = self.working_dir+"/datasets/RF00005_CCU.fasta"
            self.alphabet_type = "RNA"
            self.theta = 0.2


    def one_hot_3D(self, s):
        """ Transform sequence string into one-hot aa vector"""
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i , self.aa_dict[letter]] = 1
        return x

    def gen_basic_alignment(self):
        """ Read training alignment and store basics in class instance """
        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i:aa for aa,i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        # Read alignment
        self.seq_name_to_sequence = defaultdict(str)
        self.seq_names = []

        name = ""
        INPUT = open(self.alignment_file, "r")
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith(">"):
                name = line
                self.seq_names.append(name)
            else:
                self.seq_name_to_sequence[name] += line
        INPUT.close()

        # If we don"t have a focus sequence, pick the one that
        #   we used to generate the alignment
        if self.focus_seq_name == "":
            self.focus_seq_name = self.seq_names[0]

        # Select focus columns
        #  These columns are the uppercase residues of the .a2m file
        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper()]
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # We also expect the focus sequence to be formatted as:
        # >[NAME]/[start]-[end]
        focus_loc = self.focus_seq_name.split("/")[-1]
        start,stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_cols_list \
            = [idx_col+int(start) for idx_col in self.focus_cols]
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols}
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols}


    def gen_full_alignment(self):

        # Get only the focus columns
        for seq_name,sequence in self.seq_name_to_sequence.items():
            # Replace periods with dashes (the uppercase equivalent)
            sequence = sequence.replace(".","-")

            #then get only the focus columns
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have bad characters
        alphabet_set = set(list(self.alphabet))
        seq_names_to_remove = []
        for seq_name,sequence in self.seq_name_to_sequence.items():
            for letter in sequence:
                if letter not in alphabet_set and letter != "-":
                    seq_names_to_remove.append(seq_name)

        seq_names_to_remove = list(set(seq_names_to_remove))
        for seq_name in seq_names_to_remove:
            del self.seq_name_to_sequence[seq_name]
        print("sequence",sequence)
        # Encode the sequences
        print ("Encoding sequences")
        self.x_train = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        self.x_train_name_list = []
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            self.x_train_name_list.append(seq_name)
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    self.x_train[i,j,k] = 1.0


        # Fast sequence weights with Theano
        if self.calc_weights:
            print ("Computing sequence weights")
            # Numpy version
            # import scipy
            # from scipy.spatial.distance import pdist, squareform
            # self.weights = scale / np.sum(squareform(pdist(seq_index_array, metric="hamming")) < theta, axis=0)
            #
            # Theano weights
            X = T.tensor3("x")
            cutoff = T.scalar("theta")
            X_flat = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
            N_list, updates = theano.map(lambda x: 1.0 / T.sum(T.dot(X_flat, x) / T.dot(x, x) > 1 - cutoff), X_flat)
            weightfun = theano.function(inputs=[X, cutoff], outputs=[N_list],allow_input_downcast=True)
            #
            self.weights = weightfun(self.x_train, self.theta)[0]

        else:
            # If not using weights, use an isotropic weight matrix
            self.weights = np.ones(self.x_train.shape[0])

        self.Neff = np.sum(self.weights)

        print ("Neff =",str(self.Neff))
        print ("Data Shape =",self.x_train.shape)


    def delta_elbo(self, model, mutant_tuple_list, N_pred_iterations=10):

        for pos,wt_aa,mut_aa in mutant_tuple_list:
            if pos not in self.uniprot_focus_col_to_wt_aa_dict \
                or self.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa:
                print ("Not a valid mutant!",pos,wt_aa,mut_aa)
                return None

        mut_seq = self.focus_seq_trimmed[:]
        for pos,wt_aa,mut_aa in mutant_tuple_list:
            mut_seq[self.uniprot_focus_col_to_focus_idx[pos]] = mut_aa


        if self.wt_elbo == None:
            mutant_sequences = [self.focus_seq_trimmed, mut_seq]
        else:
            mutant_sequences = [mut_seq]

        # Then make the one hot sequence
        mutant_sequences_one_hot = np.zeros(\
            (len(mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(mutant_sequences):
            for j,letter in enumerate(sequence):
               if letter != '-' and letter != 'X' and letter !='Z':
                k = self.aa_dict[letter]
                mutant_sequences_one_hot[i,j,k] = 1.0

        prediction_matrix = np.zeros((mutant_sequences_one_hot.shape[0],N_pred_iterations))
        idx_batch = np.arange(mutant_sequences_one_hot.shape[0])
        for i in range(N_pred_iterations):

            batch_preds, _, _ = model.all_likelihood_components(mutant_sequences_one_hot)

            prediction_matrix[:,i] = batch_preds

        # Then take the mean of all my elbo samples
        mean_elbos = np.mean(prediction_matrix, axis=1).flatten().tolist()

        if self.wt_elbo == None:
            self.wt_elbo = mean_elbos.pop(0)

        return mean_elbos[0] - self.wt_elbo

    def single_mutant_matrix(self, model, N_pred_iterations=10, \
            minibatch_size=2000, filename_prefix=""):

        """ Predict the delta elbo for all single mutants """

        # Get the start and end index from the sequence name
        start_idx, end_idx = self.focus_seq_name.split("/")[-1].split("-")
        start_idx = int(start_idx)

        wt_pos_focus_idx_tuple_list = []
        focus_seq_index = 0
        focus_seq_list = []
        for i,letter in enumerate(self.focus_seq):
            if letter == letter.upper():
                wt_pos_focus_idx_tuple_list.append((letter,start_idx+i,focus_seq_index))
                focus_seq_index += 1

        self.mutant_sequences = ["".join(self.focus_seq_trimmed)]
        self.mutant_sequences_descriptor = ["wt"]
        for wt,pos,idx_focus in wt_pos_focus_idx_tuple_list:
            for mut in self.alphabet:
                if wt != mut:
                    # Make a descriptor
                    descriptor = wt+str(pos)+mut

                    # Hard copy the sequence
                    focus_seq_copy = list(self.focus_seq_trimmed)[:]

                    # Mutate
                    focus_seq_copy[idx_focus] = mut

                    # Add to the list
                    self.mutant_sequences.append("".join(focus_seq_copy))
                    self.mutant_sequences_descriptor.append(descriptor)

        # Then make the one hot sequence
        self.mutant_sequences_one_hot = np.zeros(\
            (len(self.mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(self.mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                self.mutant_sequences_one_hot[i,j,k] = 1.0

        self.prediction_matrix = np.zeros((self.mutant_sequences_one_hot.shape[0],N_pred_iterations))

        batch_order = np.arange(self.mutant_sequences_one_hot.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.mutant_sequences_one_hot.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.mutant_sequences_one_hot[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

        # Then take the mean of all my elbo samples
        self.mean_elbos = np.mean(self.prediction_matrix, axis=1).flatten().tolist()

        self.wt_elbo = self.mean_elbos.pop(0)
        self.mutant_sequences_descriptor.pop(0)

        self.delta_elbos = np.asarray(self.mean_elbos) - self.wt_elbo

        if filename_prefix == "":
            return self.mutant_sequences_descriptor, self.delta_elbos

        else:
            OUTPUT = open(filename_prefix+"_samples-"+str(N_pred_iterations)\
                +"_elbo_predictions.csv", "w")

            for i,descriptor in enumerate(self.mutant_sequences_descriptor):
                OUTPUT.write(descriptor+";"+str(self.mean_elbos[i])+"\n")

            OUTPUT.close()


    def custom_mutant_matrix(self, input_filename, model, N_pred_iterations=10, \
            minibatch_size=2000, filename_prefix="", offset=0):

        """ Predict the delta elbo for a custom mutation filename
        """
        # Get the start and end index from the sequence name
        start_idx, end_idx = self.focus_seq_name.split("/")[-1].split("-")
        start_idx = int(start_idx)

        wt_pos_focus_idx_tuple_list = []
        focus_seq_index = 0
        focus_seq_list = []
        mutant_to_letter_pos_idx_focus_list = {}

        # find all possible valid mutations that can be run with this alignment
        for i,letter in enumerate(self.focus_seq):
            if letter == letter.upper():
                for mut in self.alphabet:
                    pos = start_idx+i
                    if letter != mut:
                        mutant = letter+str(pos)+mut
                        mutant_to_letter_pos_idx_focus_list[mutant] = [letter,start_idx+i,focus_seq_index]
                focus_seq_index += 1

        '''print("----------------------------------")
        print(self.focus_seq)
        print(mutant_to_letter_pos_idx_focus_list)
        print("----------------------------------")'''

        self.mutant_sequences = ["".join(self.focus_seq_trimmed)]
        self.mutant_sequences_descriptor = ["wt"]

        # run through the input file  输入突变文库
        INPUT = open(self.working_dir+"/examples/"+input_filename, "r")
        for i,line in enumerate(INPUT):
            line = line.rstrip()
            if i >= 1:
                line_list = line.split(",")
                # generate the list of mutants
                mutant_list = line_list[0].split(":")
                valid_mutant = True

                # if any of the mutants in this list aren"t in the focus sequence,
                #    I cannot make a prediction
                for mutant in mutant_list:
                    if mutant not in mutant_to_letter_pos_idx_focus_list:
                        #self.mutant_sequences_descriptor.append(":".join(mutant_list))
                        #self.delta_elbos=0

                        valid_mutant = False

                # If it is a valid mutant, add it to my list to make preditions
                if valid_mutant:
                    focus_seq_copy = list(self.focus_seq_trimmed)[:]
                    
                    for mutant in mutant_list:
                        wt_aa,pos,idx_focus = mutant_to_letter_pos_idx_focus_list[mutant]
                        mut_aa = mutant[-1]
                        focus_seq_copy[idx_focus] = mut_aa

                    self.mutant_sequences.append("".join(focus_seq_copy))
                    self.mutant_sequences_descriptor.append(":".join(mutant_list))
        INPUT.close()

        # Then make the one hot sequence
        self.mutant_sequences_one_hot = np.zeros(\
            (len(self.mutant_sequences),len(self.focus_cols),len(self.alphabet)))

        for i,sequence in enumerate(self.mutant_sequences):
            for j,letter in enumerate(sequence):
                k = self.aa_dict[letter]
                self.mutant_sequences_one_hot[i,j,k] = 1.0

        self.prediction_matrix = np.zeros((self.mutant_sequences_one_hot.shape[0],N_pred_iterations))

        batch_order = np.arange(self.mutant_sequences_one_hot.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.mutant_sequences_one_hot.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.mutant_sequences_one_hot[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

        # Then take the mean of all my elbo samples
        self.mean_elbos = np.mean(self.prediction_matrix, axis=1).flatten().tolist()

        self.wt_elbo = self.mean_elbos.pop(0)
        self.mutant_sequences_descriptor.pop(0)

        self.delta_elbos = np.asarray(self.mean_elbos) - self.wt_elbo

        if filename_prefix == "":
            return self.mutant_sequences_descriptor, self.delta_elbos

        else:

            OUTPUT = open(filename_prefix+"_samples-"+str(N_pred_iterations)\
                +"_elbo_predictions.csv", "w")

            for i,descriptor in enumerate(self.mutant_sequences_descriptor):
                OUTPUT.write(descriptor+";"+str(self.delta_elbos[i])+"\n")

            OUTPUT.close()

    def get_pattern_activations(self, model, update_num, filename_prefix="",
                        verbose=False, minibatch_size=2000):

        activations_filename = self.working_dir+"/embeddings/"+filename_prefix+"_pattern_activations.csv"

        OUTPUT = open(activations_filename, "w")

        batch_order = np.arange(len(self.x_train_name_list))

        for i in range(0,len(self.x_train_name_list),minibatch_size):
            batch_index = batch_order[i:i+minibatch_size]
            one_hot_seqs = self.x_train[batch_index]
            batch_activation = model.get_pattern_activations(one_hot_seqs)

            for j,idx in enumerate(batch_index.tolist()):
                sample_activation = [str(val) for val in batch_activation[j].tolist()]
                sample_name = self.x_train_name_list[idx]
                out_line = [str(update_num),sample_name]+sample_activation
                if verbose:
                    print ("\t".join(out_line))
                OUTPUT.write(",".join(out_line)+"\n")

        OUTPUT.close()


    def get_embeddings(self, model, update_num, filename_prefix="",
                        verbose=False, minibatch_size=2000):
        """ Save the latent variables from all the sequences in the alignment """
        embedding_filename = self.working_dir+"/embeddings/"+filename_prefix+"_seq_embeddings.csv"

        # Append embeddings to file if it has already been created
        #   This is useful if you want to see the embeddings evolve over time
        if os.path.isfile(embedding_filename):
            OUTPUT = open(embedding_filename, "a")

        else:
            OUTPUT = open(embedding_filename, "w")
            mu_header_list = ["mu_"+str(i+1) for i in range(model.n_latent)]
            log_sigma_header_list = ["log_sigma_"+str(i+1) for i in range(model.n_latent)]

            header_list = mu_header_list + log_sigma_header_list
            OUTPUT.write("update_num,name,"+",".join(header_list)+"\n")


        batch_order = np.arange(len(self.x_train_name_list))

        for i in range(0,len(self.x_train_name_list),minibatch_size):
            batch_index = batch_order[i:i+minibatch_size]
            one_hot_seqs = self.x_train[batch_index]
            batch_mu, batch_log_sigma  = model.recognize(one_hot_seqs)

            for j,idx in enumerate(batch_index.tolist()):
                sample_mu = [str(val) for val in batch_mu[j].tolist()]
                sample_log_sigma = [str(val) for val in batch_log_sigma[j].tolist()]
                sample_name = self.x_train_name_list[idx]
                out_line = [str(update_num),sample_name]+sample_mu+sample_log_sigma
                if verbose:
                    print ("\t".join(out_line))
                OUTPUT.write(",".join(out_line)+"\n")

        OUTPUT.close()

    def get_elbo_samples(self, model, N_pred_iterations=100, minibatch_size=2000):

        self.prediction_matrix = np.zeros((self.one_hot_mut_array_with_wt.shape[0],N_pred_iterations))

        batch_order = np.arange(self.one_hot_mut_array_with_wt.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0,self.one_hot_mut_array_with_wt.shape[0],minibatch_size):

                batch_index = batch_order[j:j+minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.one_hot_mut_array_with_wt[batch_index])

                for k,idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i]= batch_preds[k]

def gen_job_string(data_params, model_params):
    """
        Generates a unique job string given data and model parameters.
        This is used later as an identifier for the
                saved model weights and figures
        Parameters
        ------------
        data_params: dictionary of parameters for the data class
        model_params: dictionary of parameters for the model class

        Returns
        ------------
        job string denoting parameters of run
    """

    written_out_vals = ["n_latent"]
    layer_num_list = ["zero","one","two","three","four"]

    encoder_architecture = []
    decoder_architecture = []

    for layer_num in layer_num_list:
        if "encode_dim_"+layer_num in model_params:
            encoder_architecture.append(model_params["encode_dim_"+layer_num])
        if "decode_dim_"+layer_num in model_params:
            decoder_architecture.append(model_params["decode_dim_"+layer_num])

        written_out_vals += ["encode_dim_"+layer_num, "decode_dim_"+layer_num]

    n_latent = model_params["n_latent"]

    encoder_architecture_str = "-".join([str(size) for size in encoder_architecture])
    decoder_architecture_str = "-".join([str(size) for size in decoder_architecture])

    job_str = "vae_output_encoder-"+encoder_architecture_str+"_Nlatent-"+str(n_latent)\
        +"_decoder-"+decoder_architecture_str

    job_id_list = []
    for data_id,data_val in sorted(data_params.items()):
        if data_id not in written_out_vals:
            if str(type(data_val)) == "<type 'list'>":
                job_id_list.append(data_id+"-"+"-".join([str(val) for val in data_val]))
            else:
                job_id_list.append(data_id+"-"+str(data_val))


    for model_id,model_val in sorted(model_params.items()):
        if model_id not in written_out_vals:
            if str(type(model_val)) == "<type 'list'>":
                job_id_list.append(model_id+"-"+"-".join([str(val) for val in model_val]))
            else:
                job_id_list.append(model_id+"-"+str(model_val))


    return job_str+"_"+"_".join(job_id_list)