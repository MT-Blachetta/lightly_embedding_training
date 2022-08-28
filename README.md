# lightly_embedding_training
Implements self-supervised model (DNN) pretraining for feature representations as embeddings
based on the framework "lightly"

# User Manual

To start the application execute one of the files with a "MAIN.py" suffix in the command line.
The application is controlled by a config file you pass as the prefix command line argument to run the python file.
The config file name ends with a ".yml" suffix. Other application settings you need to add to the command line
are the "gpu id" and the "root directory". To run a training process use a command that looks like this:

> python single_MAIN.py -gpu 0 -p moco -root_dir RESULTS

The prefix command "-p" refers to the config file name without the prefix '.yml'. In this example the application loads
the parameters in configuration file "moco.yml" into the programm. The Algorithm to use and all details for the execution
are defined by the parameters (as name and value pairs) of the config file. The specific parameters are explained here:

	 # Setup
	setup: moco # select the representation learning method (algorithm)
	train_method: moco # each algorithm has it own training_function wich contains the training loop over the dataset for one epoch
	version: moco # the implemented methods are [clpcl (our method), simclr, nnclr, simsiam, barlowtwins, byol, moco, scatSimCLR (https://github.com/vkinakh/scatsimclr)]


	 # Model
	 
	# each embedding learning method has another network architecture but all have a backbone network as component
	backbone: ResNet34 # the following backbone networks are implemented: [ResNet18,ResNet34,ResNet50, scatnet (https://github.com/vkinakh/scatsimclr)]
	
	# the full model architecture depends on the learning method
	base_model: moco # for each method, a corresponding base_model exists (contains the backbone): [scatnet,barlow,simclr,byol,nnclr,simsiam,swav,clpcl,moco]
	
	feature_dim: 128 # the dimension of the trained embeddings
	model_kwargs: # this parameters are used for the scatnet network architecture
		input_size: [96, 96, 3] 
		res_blocks: 30
		hidden_dim: 256
		out_dim: 128

	 # Dataset
	train_db_name: stl-10
	val_db_name: stl-10
	num_classes: 10
	split: train+unlabeled

	 # LossA
	temperature: 0.1
	criterion: clpcl
	criterion_kwargs:
	   temperature: 0.1 

	 # Hyperparameters
	epochs: 500
	optimizer: sgd
	optimizer_kwargs:
	   nesterov: False
	   weight_decay: 0.0001 
	   momentum: 0.9
	   lr: 0.4
	scheduler: cosine
	scheduler_kwargs:
	   lr_decay_rate: 0.1
	batch_size: 512 
	num_workers: 8

	 # Transformations
	augmentation_strategy: multicrop 
	augmentation_kwargs:
	   local_crops_number: 4
	   local_crops_scale: [0.05, 0.4]
	   local_crops_size: 96
	   global_crops_scale: [0.4, 1.0]
	   random_resized_crop:
		  size: 96
		  scale: [0.2, 1.0]
	   color_jitter_random_apply:
		  p: 0.8
	   color_jitter:
		  brightness: 0.4
		  contrast: 0.4
		  saturation: 0.4
		  hue: 0.1
	   random_grayscale: 
		  p: 0.2
	   random_horizontal_flip: 0.5
	   gaussian_blur: [0.1,2.0]
	   Solarization: 0.2
	   normalize:
		  mean: [0.485, 0.456, 0.406]
		  std: [0.229, 0.224, 0.225]

	transformation_kwargs:
	   crop_size: 96
	   normalize:
		  mean: [0.485, 0.456, 0.406]
		  std: [0.229, 0.224, 0.225]
