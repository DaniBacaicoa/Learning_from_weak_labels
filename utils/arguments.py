import argparse

def argument_parser():
    #First, i include a function so it decides wether to return a number or a list of numbers
    def arg_type(value,data_type):
        try:
            value = data_type(value)
            return value
        except ValueError:
            values = value.split()
            return [data_type(v) for v in values]

    # main setting
    parser = argparse.ArgumentParser(description='Arguments for Learning with Weakly labeled datasets')


    #Dataset arguments
    parser.add_argument('--ds', help = 'Dataset to be used', type = str, default = 'mnist')
    parser.add_argument('--bs', help = 'Batch size for training', type = int, default = 32)
    parser.add_argument('--ep', help = 'Number of training epochs', type = int, default = 10)
    parser.add_argument('--warm', help='Number of warm_up epochs', type = int, default = 10)

    #MLP arguments
    parser.add_argument('--dropout', help = 'Dropout probability', type = float, default = 0.)
    parser.add_argument('--hidden', help = 'Number of neurons in ech hidden layer',
                        type = lambda x: arg_type(x,int), nargs='*', default = [])

    #Weakening and reconstruction arguments
    parser.add_argument('--weakening', help = 'Weakening procedure', type = str, default = 'pll',
                        choices = ['pu','supervised','noisy','complementary','weak','pll','pll_a'])
    parser.add_argument('--virtual', help='Reconstruction procedure', type=str, default='M-opt-conv',
                        choices=['Inv', 'M-opt', 'M-conv', 'M-opt-conv'])
    parser.add_argument('--alpha', help = 'Weakening parameter(-1,infty):Weakening for each label.(noisy,weak)',
                        type = lambda x: arg_type(x,float), nargs='*', default = 0.5)
    parser.add_argument('--pll_p', help = 'Weakening parameter(0,1): Probability of flipping coin.(pll,pll_a)', type = float, default = 0.5)

    #Losses arguments
    parser.add_argument('--loss', help='Loss function', type=str, default='CELoss',
                        choices=['CELoss', 'BrierLoss', 'EMLoss', 'PartialLoss', 'LBLoss', 'OSLCELoss', 'OSLBrierLoss'])
    parser.add_argument('--optim', help='Algorithm from torch.optim', type=str, default='SGD',
                        choices=['SGD', 'Adam', 'Adagrad', 'Adadelta', 'LBFGS', 'RMSprop', 'OSLBrierLoss'])
    parser.add_argument('--optim_params', help='Parameters for the optimization algorithm', type=dict, default={"lr": 0.01})
    #parser.add_argument('--optim_params', type=dict, default={"lr": 0.01,"weight_decay":1e-4, "momentum": 0.5})
    parser.add_argument('--lbl_params', help='Parameters for the LBLoss', type=dict, default={"k": 0.5,"beta":2})

    #Results saving arguments
    parser.add_argument('--dir', help='result save path', type=str, default='results/', required=False)

    args = parser.parse_args()
    return args