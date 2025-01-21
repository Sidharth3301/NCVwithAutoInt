import torch
class control_variate_loss(torch.nn.Module):
    def __init__(self, U, L):
        super(control_variate_loss, self).__init__()
        self.error_fn = torch.nn.MSELoss()
        self.U = U
        self.L = L
        # self.func_to_fit = func_to_fit
        self.alpha =1e-3

    def l_diff(self, gradnet_outputs, gt):
        return 2*self.error_fn( gradnet_outputs, gt).mean() #L_diff for uniform sampling : the 2 comes from the denominator p_\omega^2
    
    def l_int(self, integralnet_outputs, gt):
        a,b = integralnet_outputs[0], integralnet_outputs[1]
        integral = a-b
        term1 = gt*(self.U-self.L)
        return self.error_fn(integral, term1).mean()
        
    
    def forward(self, integralnet_outputs, gradnet_outputs, gt):
        return self.l_diff(gradnet_outputs=gradnet_outputs, gt=gt)+self.alpha*self.l_int(integralnet_outputs=integralnet_outputs, gt=gt)
    
    

class autoInt_loss(torch.nn.Module):
    def __init__(self):
        super(autoInt_loss, self).__init__()
        self.error_fn = torch.nn.MSELoss()
        # self.U = U
        # self.L = L
        # self.func_to_fit = func_to_fit
        # self.alpha =0.99

    def l_diff(self, gradnet_outputs, gt):
        return self.error_fn(gradnet_outputs, gt).mean() #L_diff for uniform sampling : the 2 comes from the denominator p_\omega^2
    
    # def l_int(self, integralnet_outputs, gt):
    #     a,b = integralnet_outputs[0], integralnet_outputs[1]
    #     integral = a-b
    #     term1 = gt*(self.U-self.L)
    #     return self.error_fn(integral, term1).mean()
        
    
    def forward(self, gradnet_outputs, gt):
        return self.l_diff(gradnet_outputs=gradnet_outputs, gt=gt)