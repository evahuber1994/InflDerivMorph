#!/bin/bash
for hl in 1 2 3; do
	for non_l in True False; do
		for loss in mse cosine_distance; do
			name="results/russian_$hl+$non_l+$loss"
			echo $name 
			mkdir $name
			mkdir $name/model
			Config_File="$name/toml_RU.toml"

			echo train_path = \"/home/eva/master_diss/ru_conll_data_2nd/combined_train.csv\" >> $Config_File
			echo val_path =  \"/home/eva/master_diss/ru_conll_data_2nd/combined_val.csv\" >> $Config_File
			echo test_path = \"/home/eva/master_diss/ru_conll_data_2nd/combined_test.csv\" >> $Config_File
			echo out_path = \"$name/\" >> $Config_File
			echo model_path = \"$name/model/model\" >> $Config_File
			echo save_detailed = \"True\" >> $Config_File
			echo embeddings = \"/home/eva/master_diss/ru_conll_data/model.fifu\" >> $Config_File
			echo embedding_dim = 100 >> $Config_File 
			echo rel_embedding_dim = 100 >> $Config_File 
			echo restricted_vocabulary_matrix = \"True\" >> $Config_File
			echo nr_epochs = 100 >> $Config_File
			echo patience = 10 >> $Config_File
			echo batch_size = 24 >> $Config_File
			echo dropout_rate = 0.0 >> $Config_File
			echo loss_type = \"$loss\" >> $Config_File
			echo non_linearity = \"$non_l\" >> $Config_File
			echo non_linearity_function = \"sigmoid\" >> $Config_File
			echo nr_layers = $hl >> $Config_File
			echo hidden_dim = 100 >> $Config_File
			echo device = \"cuda\" >> $Config_File
			echo early_stopping_criterion = \"cosine_similarity\" >> $Config_File

			echo written config file 
			python3 main_relations.py $Config_File
		done
	done
done

