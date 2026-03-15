import torch
import numpy as np
from torch.nn.modules.loss import _Loss


class DistCor(_Loss):
    def __init__(self):
        super(DistCor, self).__init__()

    def Distance_Correlation(self, channel1, channel2):
        assert channel1.shape[1] == channel2.shape[1], "The dim of two tensors need to be equal"
        dim_num = channel1.shape[1]
        correlation_r = 0
        for i in range(dim_num):
            latent = channel1[:, i]
            latent = latent.unsqueeze(1)
            control = channel2[:, i]
            control = control.unsqueeze(1)

            matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
            matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)

            matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
            matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

            Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
            Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
            Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

            correlation_r += Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r

    def forward(self, channel1, channel2):
        dc_loss = self.Distance_Correlation(channel1, channel2)
        return dc_loss


class Results:
    def __init__(self, seed_num, model_num):
        super(Results, self).__init__()

        self.seed_num = seed_num
        self.model_num = model_num
        self.auc, self.f1, self.acc, self.parity, self.equality = np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num))

    def report_results(self):
        for i in range(self.model_num):
            print(f"============" + "START" + "+" + f"model_{i}" + "============")
            print(f"AUCROC: {np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}")
            print(f'F1-score: {np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}')
            print(f'ACC: {np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}')
            print(f'Parity: {np.around(np.mean(self.parity[:, i]) * 100, 2)} ± {np.around(np.std(self.parity[:, i]) * 100, 2)}')
            print(f'Equality: {np.around(np.mean(self.equality[:, i]) * 100, 2)} ± {np.around(np.std(self.equality[:, i]) * 100, 2)}')
            print("=================END=================")

    def save_results(self, args):
        for i in range(self.model_num):
            with open(f"{args.dataset}_tao{args.tem}_model{i}.txt", 'a') as f:
                f.write(f"τ={args.tem}, γ={args.alpha}, lr_w={args.lr_w}\n")
                f.write(f"AUCROC: {np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}\n")
                f.write(f'F1-score: {np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}\n')
                f.write(f'ACC: {np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}\n')
                f.write(f'Parity: {np.around(np.mean(self.parity[:, i]) * 100, 2)} ± {np.around(np.std(self.parity[:, i]) * 100, 2)}\n')
                f.write(f'Equality: {np.around(np.mean(self.equality[:, i]) * 100, 2)} ± {np.around(np.std(self.equality[:, i]) * 100, 2)}\n')
            f.close()