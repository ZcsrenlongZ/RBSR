class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = '/hdd/burst_SR/deep-burst-sr-master-L1-BasicVSRpp-shiyanshijiqun/pretrained_networks'    # Directory for pre-trained networks.
        self.save_data_path = self.workspace_dir + '/evaluation'    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = '../dataset/Zurich-RAW-to-DSLR-Dataset'    # Zurich RAW 2 RGB path
        self.burstsr_dir = '../dataset/burstsr_dataset'    # BurstSR dataset path
        self.synburstval_dir = '../dataset/syn_burst_val'    # SyntheticBurst validation set path
