from args import parse_train_opt
from EDGE import EDGE
import jittor as jt

jt.flags.use_cuda=1

def train(opt):
    model = EDGE(opt.feature_type)
    model.train_loop(opt)

if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
    