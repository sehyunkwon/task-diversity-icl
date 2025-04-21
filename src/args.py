import argparse
import json


MODELS = ['san', 'lstm', 'gpt', 'llama', 'gru', 'mysan', 'attclf', 'retnet', 'hyena', 'dss']
MODEL_LIST = ["gpt2", "gpt2-large", "gpt2-xl", 'lin_attn', 'llama7b']

def build_parser():	
	parser = argparse.ArgumentParser(description='Run')
	parser.add_argument('-save_iteration', type=int, default=None, help='Saving iteration for one task')
	parser.add_argument('-pretrained', type=str, default=None, help='Load pretrained model')
	parser.add_argument('-json_file_path', type=str, help='Path to the JSON config file', default="")
	parser.add_argument('-task_list', type=list, help='', default=[])
	parser.add_argument('-split_ratio', type=str, help='Split train set and test set. Ex, if split_ratio=0.8 -> train:test = 0.8:0.2')
	parser.add_argument('-ncl_bsize', type=int, default=2**10, help='ncl_bsize')
	parser.add_argument('-ncl_wsize', type=int, default=2**10, help='ncl_wsize')

	# Miscellaneous
	parser.add_argument('-wandb', dest='wandb', action='store_true', help='Store wandb')
	parser.add_argument('-no-wandb', dest='wandb', action='store_false', help='Do not store wandb')
	parser.set_defaults(wandb=False)
	parser.add_argument('-gpu', type=int, nargs='+', default=[0, 1], help="Select list of gpus. e.g., 2 3 4")

	parser.add_argument('-out_dir', type=str, default='./models/', help='outputs directory')
	parser.add_argument('-delete', dest='delete', action='store_true', help='delete model after run')
	parser.add_argument('-no-delete', dest='delete', action='store_false', help='Do not delete model after run')
	parser.set_defaults(delete=True)

	parser.add_argument('-test_run', dest='test_run', action='store_true', help='Test run')
	parser.add_argument('-no-test_run', dest='test_run', action='store_false', help='Not test run')
	parser.set_defaults(test_run=False)

	# Model
	parser.add_argument('-family', type=str, default='mysan', choices= MODELS,  help='Model Family')
	parser.add_argument('-model_name', type=str, default='gpt2', choices= MODEL_LIST,  help='Select Pretrained Model')
	parser.add_argument('-llama_weights_path', type=str, default='/home/', help='Outer directory where hf converted LLaMA weights are stored')
	parser.add_argument('-precision', type=str, default='full', choices= ['half', 'full'],  help='Select precision for llama weights')
	parser.add_argument('-n_positions', type=int, default=150, help='Maximum context length')
	parser.add_argument('-n_dims', type=int, default=28, help='input dimension')
	parser.add_argument('-n_embd', type=int, default=256, help='embedding dimension')
	parser.add_argument('-n_layer', type=int, default=6, help='number of layers')
	parser.add_argument('-n_head', type=int, default=8, help='number of heads')
	parser.add_argument('-order', type=int, default=3, help='Order: For Hyena')
	parser.add_argument('-freeze', type=int, default=0, choices= [0, 1, 2], help='0: no freeze, 1: freeze partial, 2: freeze all')

	# Training
	parser.add_argument('-optimizer', type=str, help='sgd or adam')
	parser.add_argument('-weight_decay', action='store_true', help='Enable l2 regularization')
	# parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('-learning_rate', type=float, default=0.0001, help='Learning rate')
	parser.add_argument('-train_steps', type=int, default=20001, help='number of train steps')
	parser.add_argument('-save_every_steps', type=int, default=20000, help='how often to checkpoint')
	parser.add_argument('-keep_every_steps', type=int, default=100000, help='permanent checkpoints')
	parser.add_argument('-resume_id', type=str, default='',  help='resume id')
	parser.add_argument('-analyze', dest='analyze', action='store_true', help='analyze')
	parser.add_argument('-no-analyze', dest='analyze', action='store_false', help='Do not  analyze')
	parser.set_defaults(analyze=False)

	#################################################
	# We do not use Curriculum Learning. Therefore, 
	# curriculum_dims_start = curriculum_dims_end`
	# curriculum_points_start = curriculum_points_end
	##################################################
	parser.add_argument('-curriculum_dims_start', type=int, default=28, help='initial parameter')
	parser.add_argument('-curriculum_dims_end', type=int, default=28, help='limit of final value')
	parser.add_argument('-curriculum_dims_inc', type=int, default=1, help='how much to increment each time')
	parser.add_argument('-curriculum_dims_interval', type=int, default=2000, help='increment every how many steps')
	parser.add_argument('-curriculum_points_start', type=int, default=120, help='initial parameter')
	parser.add_argument('-curriculum_points_end', type=int, default=120, help='limit of final value')
	parser.add_argument('-curriculum_points_inc', type=int, default=5, help='how much to increment each time')
	parser.add_argument('-curriculum_points_interval', type=int, default=2000, help='increment every how many steps')

	#################################################
	# We do not use Prefix Scoring. Therefore,
	# parser.set_defaults(prefix_score_train=False)
	# parser.set_defaults(prefix_score_eval=False)
	##################################################
	parser.add_argument('-prefix_score_train', dest='prefix_score_train', action='store_true', help='Calculate prefix scores during training')
	parser.add_argument('-no-prefix_score_train', dest='prefix_score_train', action='store_false', help='Do not calculate prefix scores during training')
	parser.set_defaults(prefix_score_train=False)
	parser.add_argument('-prefix_score_eval', dest='prefix_score_eval', action='store_true', help='Calculate prefix scores during evaluation')
	parser.add_argument('-no-prefix_score_eval', dest='prefix_score_eval', action='store_false', help='Do not calculate prefix scores during evaluation')
	parser.set_defaults(prefix_score_eval=False)
	parser.add_argument('-prefix_score_train_interval', type=int, default=500, help='calculate prefix score after every how many steps')
	parser.add_argument('-prefix_score_n_repeats', type=int, default=4, help='How many times to repeat sequence')
	parser.add_argument('-prefix_score_n_points', type=int, default=12, help='How many points in sequence')
	parser.add_argument('-prefix_score_bsize', type=int, default=100, help='Average score over how many examples - should not be more than bsize')

	# nearest neighbours Scoring
	parser.add_argument('-nn_score_train', dest='nn_score_train', action='store_true', help='Calculate nn scores during training')
	parser.add_argument('-no-nn_score_train', dest='nn_score_train', action='store_false', help='Do not calculate nn scores during training')
	parser.set_defaults(nn_score_train=False)
	parser.add_argument('-nn_score_eval', dest='nn_score_eval', action='store_true', help='Calculate nn scores during evaluation')
	parser.add_argument('-no-nn_score_eval', dest='nn_score_eval', action='store_false', help='Do not calculate nn scores during evaluation')
	parser.set_defaults(nn_score_eval=False)
	parser.add_argument('-nn_score_train_interval', type=int, default=500, help='calculate nn score after every how many steps')
	parser.add_argument('-nn_score_n_points', type=int, default=70, help='How many points in sequence')
	parser.add_argument('-nn_score_start_point', type=int, default=41, help='From which point to start nn scoring')
	parser.add_argument('-nn_score_bsize', type=int, default=100, help='Average score over how many examples - should not be more than bsize')
	parser.add_argument('-nn_score_num_neighbours', type=int, default=1, help='How many nearest neighbours to observe for attn weights?')

	# Wandb
	parser.add_argument('-project', type=str, default='in-context-test', help='wandb project name')
	parser.add_argument('-entity', type=str, default='name', help='wandb entity name')
	parser.add_argument('-notes', type=str, default='', help='wandb notes')
	parser.add_argument('-name', type=str, default='conj_test', help='run name')
	parser.add_argument('-log_every_steps', type=int, default=10, help='wandb log every how many steps')
	


	return parser
