#!/bin/bash
for i in {1..10}; do
	name="results/turkish_multiruns/turkish_$i"
	mkdir $name 
	mkdir $name/model
	echo $name 
	Config_File="$name/toml_TUR_multr.toml"

	echo train_path = \"/home/eva/master_diss/FINAL/TUR/normal_o1o/combined_train.csv\" >> $Config_File
	echo val_path =  \"/home/eva/master_diss/FINAL/TUR/normal_o1o/combined_val.csv\" >> $Config_File
	echo test_path = \"/home/eva/master_diss/FINAL/TUR/normal_o1o/combined_test.csv\" >> $Config_File
	echo out_path = \"$name/\" >> $Config_File
	echo model_path = \"$name/model/model\" >> $Config_File
	echo save_detailed = \"True\" >> $Config_File
	echo embeddings = \"/home/eva/master_diss/tur_conll_data/model.fifu\" >> $Config_File
	echo embedding_dim = 100 >> $Config_File 
	echo rel_embedding_dim = 100 >> $Config_File 
	echo restricted_vocabulary_matrix = \"True\" >> $Config_File
	echo nr_epochs = 100 >> $Config_File
	echo patience = 10 >> $Config_File
	echo batch_size = 24 >> $Config_File
	echo dropout_rate = 0.0 >> $Config_File
	echo loss_type = \"cosine_distance\" >> $Config_File
	echo non_linearity = \"True\" >> $Config_File
	echo non_linearity_function = \"sigmoid\" >> $Config_File
	echo nr_layers = 3 >> $Config_File
	echo hidden_dim = 100 >> $Config_File
	echo device = \"cuda\" >> $Config_File
	echo early_stopping_criterion = \"cosine_similarity\" >> $Config_File

	echo written config file 
	python3 main_relations.py $Config_File	

done

