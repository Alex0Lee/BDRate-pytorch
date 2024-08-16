import torch
from torch import nn


class BDrate(nn.Module):
    def __init__(self, device="cuda"):
        super(BDrate, self).__init__()
        self.device = device
    
    def forward(self, bitrate, psnr, bitrate_ref, psnr_ref):
        """
        Calculate and return the quality difference metric between two video streams.
        
        Parameters:
        - bitrate: The bitrate of the current video stream.
        - psnr: The Peak Signal-to-Noise Ratio (PSNR) of the current video stream.
        - bitrate_ref: The bitrate of the reference video stream.
        - psnr_ref: The Peak Signal-to-Noise Ratio (PSNR) of the reference video stream.
        
        Returns:
        - avg_diff: The quality difference metric between the two video streams.
        
        计算并返回两个视频流的质量差异指标。
        
        参数:
        - bitrate: 当前视频流的比特率
        - psnr: 当前视频流的峰值信噪比（PSNR）
        - bitrate_ref: 参考视频流的比特率
        - psnr_ref: 参考视频流的峰值信噪比（PSNR）
        
        返回:
        - avg_diff: 两个视频流的质量差异指标
        """
        # Convert all input variables to double precision and move them to the specified device (CPU or GPU) for accuracy and efficiency
        # 将所有输入变量转换为double类型并移动到指定的设备上（CPU或GPU），以确保计算的准确性和效率
        bitrate = bitrate.double().to(self.device)
        psnr = psnr.double().to(self.device)
        bitrate_ref = bitrate_ref.double().to(self.device)
        psnr_ref = psnr_ref.double().to(self.device)
        
        # Calculate the logarithm of the bitrates for subsequent calculations
        # 计算比特率的对数，用于后续的计算
        lR1 = torch.log(bitrate)
        lR2 = torch.log(bitrate_ref)
        
        # Use the polynomial fitting formula to compute the curves for both the current and reference video streams
        # 使用多项式拟合公式分别计算当前视频流和参考视频流的拟合曲线
        p1 = self.n_ploy_fit_formula(psnr, lR1)
        p2 = self.n_ploy_fit_formula(psnr_ref, lR2)
        
        # Determine the minimum and maximum intervals of PSNR values for both video streams
        # 计算两个视频流PSNR值的最小和最大区间，用于计算面积差
        min_int = torch.max(torch.min(psnr), torch.min(psnr_ref))
        max_int = torch.min(torch.max(psnr), torch.max(psnr_ref))
        
        # Compute the area under the curve for the current video stream within the PSNR interval
        # 计算当前视频流在PSNR最大和最小值之间的拟合曲线下的面积
        int1 = self.polyint(p1, max_int) - self.polyint(p1, min_int)
        
        # Compute the area under the curve for the reference video stream within the PSNR interval
        # 计算参考视频流在PSNR最大和最小值之间的拟合曲线下的面积
        int2 = self.polyint(p2, max_int) - self.polyint(p2, min_int)
        
        # Calculate the average value of the difference in areas under the curves
        # 计算两个视频流拟合曲线下的面积差的平均值
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        avg_diff = (torch.exp(avg_exp_diff) - 1)
        
        # Return the quality difference metric
        # 返回两个视频流的质量差异指标
        return avg_diff.float()
        
        
    
    def n_ploy_fit_formula(self, x, y):
        """
        N次多项式拟合, 公式法
        N-th order polynomial fitting using the formula method
        
        :param x: 输入变量x
        :param x: Input variable x
        :param y: 输入变量y
        :param y: Input variable y
        :return: 多项式拟合的系数
        :return: Coefficients of the polynomial fit
        """
        # 构造特征矩阵X，包括x的三次方，平方，一次方以及常数项
        # Construct the feature matrix X including x^3, x^2, x, and a constant term
        X = torch.stack([x**3, x**2, x, torch.ones_like(x)], dim=1)

        # 计算 X 的转置
        # Compute the transpose of X
        Xt = X.t()

        # 应用正规方程求解系数: (Xt * X)^(-1) * Xt * y
        # Apply the normal equation to solve for coefficients: (Xt * X)^(-1) * Xt * y
        try:
            coefficients = torch.inverse(Xt @ X) @ Xt @ y
            return coefficients
        except:
            print("Error in n_ploy_fit_formula occurred")
            print("Xt @ X: ", Xt @ X)
            print("Xt: ", Xt)
            print("X: ", X)
            print("y: ", y)
            exit()
        
    
    
    def polyint(self, w, x):
        """
        Integrate a polynomial.
        
        对多项式进行积分。

        Parameters:
        w : torch.Tensor
            Weights vector, used to weight and sum the polynomial terms.
            权重向量，用于对多项式进行加权求和。
        x : float
            Variable value in the polynomial to be integrated.
            积分多项式中的变量值。
            
        Returns:
        float
            The result of the polynomial integration.
            多项式积分的结果。
        """
        # Initialize a tensor to store the integral coefficients of the polynomial
        # 初始化一个张量，存储多项式的积分系数
        X = torch.tensor([(x**4) / 4, (x**3) / 3, (x**2) / 2, x], dtype=torch.float64).to(self.device)
        # Use matrix multiplication to implement weighted summation and return the result
        # 使用矩阵乘法实现加权求和，并返回结果
        return torch.mm(w.unsqueeze(0), X.unsqueeze(-1)).item()
        
    

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
    
    def forward(self, mse):
        return 10 * torch.log10(1 / mse)
    
    
    
