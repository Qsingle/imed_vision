# -*- coding:utf-8 -*-
"""
    FileName: task_fusion
    Author: 12718
    Create Time: 2023-06-21 16:43
"""
import torch

class TargetEnhancedLoss(torch.nn.Module):
    def __init__(self):
        super(TargetEnhancedLoss,self).__init__()

    def forward(self,feature1, feature2, gt):
        fusion=feature1*feature2
        return torch.nn.MSELoss()(fusion,gt)

class TaskFusionLoss(torch.nn.Module):
    def __init__(self,subscale=0.1):
        self.subscale=int(1/subscale)
        super(TaskFusionLoss,self).__init__()

    def forward(self,feature1, feature2, gt):
        mseloss=TargetEnhancedLoss()(feature1,feature2,gt)

        feature1=torch.nn.AvgPool3d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool3d(self.subscale)(feature2)
        gt=torch.nn.AvgPool3d(self.subscale)(gt)

        fusion=feature1*feature2

        m_batchsize, C, depth, height, width = fusion.size()
        fusion = fusion.view(m_batchsize, -1, depth*width*height)  #[N,C,D*W*H]
        mat1 = torch.bmm(fusion.permute(0,2,1),fusion) #[N,D*W*H,D*W*H]

        m_batchsize, C, depth, height, width = gt.size()
        gt = gt.view(m_batchsize, -1, depth*width*height)  #[N,C,D*W*H]
        mat2 = torch.bmm(gt.permute(0,2,1),gt) #[N,D*W*H,D*W*H]

        L1norm=torch.norm(mat2-mat1,1)

        return mseloss+L1norm/((height*width)**2)