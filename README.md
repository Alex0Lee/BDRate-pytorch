# BDRate-pytorch

A torch implementation refering to the numpy implementation[ https://github.com/liyongjiandegithub/BDrate-scripts ]. The bdrate calculation is implemented using torch, with the fitting step using the least squares method and the integration step using the Newton Leibniz integration method.<br>
Compared to previous implementations, the error is in the order of 1e-6.<br>
Note: This implementation calculates the value of bdbr within [0,1], NOT as a percentage.
