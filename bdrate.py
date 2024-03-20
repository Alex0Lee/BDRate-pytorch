import torch
from torch import nn


class BDrate(nn.Module):
    def __init__(self, device="cuda"):
        super(BDrate, self).__init__()
        self.device = device
    
    def forward(self, bitrate, psnr, bitrate_ref, psnr_ref):
        # (bitrate, psnr, bitrate_ref, psnr_ref) = (bitrate.double(), psnr.double(), bitrate_ref.double(), psnr_ref.double())
        # to double device
        bitrate = bitrate.double().to(self.device)
        psnr = psnr.double().to(self.device)
        bitrate_ref = bitrate_ref.double().to(self.device)
        psnr_ref = psnr_ref.double().to(self.device)
        lR1 = torch.log(bitrate)
        lR2 = torch.log(bitrate_ref)
        
        p1 = self.n_ploy_fit_formula(psnr, lR1)
        p2 = self.n_ploy_fit_formula(psnr_ref, lR2)

        min_int = torch.max(torch.min(psnr), torch.min(psnr_ref))
        max_int = torch.min(torch.max(psnr), torch.max(psnr_ref))
        
        int1 = self.polyint(p1, max_int) - self.polyint(p1, min_int)
        
        
        int2 = self.polyint(p2, max_int) - self.polyint(p2, min_int)

        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        avg_diff = (torch.exp(avg_exp_diff)-1)
        
        return avg_diff.float()
        
        
    
    def n_ploy_fit_formula(self, x, y):
        """
        n次多项式拟合,公式法
        :return:
        """
        X = torch.stack([x**3, x**2, x, torch.ones_like(x)], dim=1)

        # 计算 X 的转置
        Xt = X.t()

        # 应用正规方程求解系数: (Xt * X)^(-1) * Xt * y
        try:
            coefficients = torch.inverse(Xt @ X) @ Xt @ y
            return coefficients
        except:
            print("error in n_ploy_fit_formula occurred")
            print("Xt @ X: ", Xt @ X)
            print("Xt: ", Xt)
            print("X: ", X)
            print("y: ", y)
            exit()
        
    
    
    def polyint(self, w, x):
        """
        Integrate a polynomial.
        """
        # X = [x**3, x**2, x, 1]

        X = torch.tensor([(x**4) / 4, (x**3) / 3, (x**2) / 2, x], dtype=torch.float64).to(self.device)
        return torch.mm(w.unsqueeze(0), X.unsqueeze(-1)).item()
        
    

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
    
    def forward(self, mse):
        return 10 * torch.log10(1 / mse)
    
    
    
