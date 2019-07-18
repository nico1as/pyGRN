from pygrn import grns, problems, evolution, config
import argparse
import os
import gym_jsbsim
import gym

parser = argparse.ArgumentParser(description='Evolve a GRN for GYM env')

parser.add_argument('--root_dir', type=str, help='Root directory', default='./')
parser.add_argument('--no-learn', dest='learn', action='store_const',  const=False, default=True, help='Turn off learning')
parser.add_argument('--no-evo', dest='evo', action='store_const', const=False, default=True, help='Turn off evolution')
parser.add_argument('--lamarckian', dest='lamarckian', action='store_const', const=True, default=False, help='Lamarckian evolution')
parser.add_argument('--unsupervised', dest='unsupervised', action='store_const', const=True, default=False, help='Unsupervised evolution')
parser.add_argument('--stateful', dest='stateful', action='store_const', const=True, default=False, help='Stateful model')
parser.add_argument('--id', type=str, help='Run id for logging')
parser.add_argument('--model', type=str, help='Model')
parser.add_argument('--ntrain', type=int, default=6*24*60, help='Number of training samples')
parser.add_argument('--ntest', type=int, default=24*60, help='Number of testing samples')
parser.add_argument('--shift', type=int, default=1, help='Data shift')
parser.add_argument('--lag', type=int, default=60, help='Time step for prediction')
parser.add_argument('--nreg', type=int, default=10, help='Number of starting regulatory proteins')
#parser.add_argument('--nout', type=int, default=1, help='Number of outputs proteins, unit size for LSTM or SimpleRNN')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--seed', type=int, help='Random seed', default=0)
parser.add_argument('--problem', type=str, help='Problem', default='GymGRN')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--gens', type=int, help='Number of generations', default=50000)
parser.add_argument('--grn_file', type=bool, help='Experts from GRN file', default=False)
parser.add_argument('--env', type=str, help='GYM env', default="GymJsbsim-HeadingControlTask-A320-v0")
parser.add_argument('--nin', type=int, help='GYM number of input', default=3)
parser.add_argument('--nout', type=int, help='GYM number of action', default=8)
parser.add_argument('--ep_max', type=int, help='episode max per loop', default=1)

parser.add_argument('--de', type=bool, help='debug', default=False)
parser.add_argument('--deo', type=bool, help='debug output', default=False)

args = parser.parse_args()

log_dir = os.path.join(args.root_dir, 'logs')

config.START_REGULATORY_SIZE = args.nreg
PROBLEM = "GymGRN"


log_file = os.path.join(log_dir, 'fits_' + args.id + '.log')
grn_dir = os.path.join(args.root_dir, 'grns')
data_dir = os.path.join(args.root_dir, 'data')
log_file = os.path.join(log_dir, 'fits_' + args.id + '.log')



p = eval('problems.' + args.problem)
p = p(args.env, args.nin, args.nout, args.ep_max)

print("..Init GRN..")
newgrn = grns.ClassicGRN() #DiffGRN()
newgrn2 = lambda: grns.ClassicGRN() #DiffGRN()
if args.evo:
	print("..Evolv GRN..")
	grneat = evolution.Evolution(p, newgrn2, run_id=args.id, grn_dir=grn_dir, log_dir=log_dir)
	grneat.run(args.gens)
else:
    for i in range(1):
        print("..Init GRN", i, "..")
        #grn = grns.ClassicGRN()
        if args.grn_file:
        	print("..Load GRN..")
        	with open(grn_dir+"/grns_"+str(args.id)+".log", 'r') as f:
        		grns = f.readlines()
        		#print("\t ", grns[-1])
        		newgrn.from_str(grns[-1])
        else:
        	print("..Gen Random GRN..")
        	newgrn.random(p.nin, p.nout, args.nreg)
        #p.generation_function(None, i)
		# print("..Eval GRN..")
        print(p.eval(newgrn, args.de, args.deo))