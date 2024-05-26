from trainer.gmd_trainer import GMDTrainer
from trainer.sim_trainer import SimTrainer
from trainer.recons_trainer import ReconsTrainer
from trainer.trainer import Trainer
from trainer.weighted_trainer import WeightedTrainer

def build_trainer(args):
    
    if args.loss.name == 'feature-sim':
        return SimTrainer(args)
    else:
        if args.trainer.name == 'feature-sim':
            return SimTrainer(args)
        elif args.trainer.name == 'recons':
            return ReconsTrainer(args)
        elif args.trainer.name == 'trainer':
            return Trainer(args)
        elif args.trainer.name == 'gmd':
            return GMDTrainer(args)
        elif args.trainer.name == 'weighted':
            return WeightedTrainer(args)
        else:
            raise NotImplementedError("Trainer not implemented!")