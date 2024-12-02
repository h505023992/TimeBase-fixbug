import os
import torch
import numpy as np
import time
from thop import profile

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.orthogonal_weight = args.orthogonal_weight
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # input = torch.randn(1,args.seq_len ,args.enc_in ).to(self.device)

        # macs, params = profile(self.model, inputs=(input, ))
        # print(f"MACs: {macs}")
        # print(f"Params: {params}")
        # if macs >= 1e9:
        #     print( f"{macs / 1e9:.2f}G MACs")
        # elif macs >= 1e6:
        #     print( f"{macs / 1e6:.2f}M MACs")
        # else:
        #     print( f"{macs} MACs")
        # # 设置运行次数
        # num_runs = 100

        # # 记录每次推理的时间
        # inference_times = []



        # input = torch.randn(1,args.seq_len ,args.enc_in )
        # model = self.model.to('cpu')
        # # 进行多次推理并记录时间
        # # 将模型设置为评估模式
        # model.eval()
        # for run in range(num_runs):
        #     start_time = time.time()
            
        #     # 进行推理
        #     with torch.no_grad():  # 禁用梯度计算，提升推理速度
        #         output = model(input)
            
        #     end_time = time.time()
        #     inference_time_s = end_time - start_time
        #     inference_time_ms = inference_time_s * 1000  # 转换为毫秒
            
        #     # 打印每次推理的时间（秒和毫秒）
        #     print(f"Run {run + 1}: Inference Time = {inference_time_s:.6f} seconds = {inference_time_ms:.2f} milliseconds")
            
        #     inference_times.append(inference_time_s)

        # # 计算平均推理时间
        # average_inference_time_s = sum(inference_times) / num_runs  # 以秒为单位
        # average_inference_time_ms = average_inference_time_s * 1000  # 转换为毫秒

        # # 输出平均推理时间
        # print(f"\nAverage Inference Time: {average_inference_time_s:.6f} seconds")
        # print(f"Average Inference Time: {average_inference_time_ms:.2f} milliseconds")
        # self.model = self.model.to(self.device)

        # # 设置运行次数
        # num_runs = 100

        # # 记录每次推理的时间
        # inference_times = []

        # print('-'*100)
        # print('gpu:')
        # input = torch.randn(1,args.seq_len ,args.enc_in ).to(self.device)
        # model = self.model
        # # 进行多次推理并记录时间
        # # 将模型设置为评估模式
        # model.eval()
        # for run in range(num_runs):
        #     start_time = time.time()
            
        #     # 进行推理
        #     with torch.no_grad():  # 禁用梯度计算，提升推理速度
        #         output = model(input)
            
        #     end_time = time.time()
        #     inference_time_s = end_time - start_time
        #     inference_time_ms = inference_time_s * 1000  # 转换为毫秒
            
        #     # 打印每次推理的时间（秒和毫秒）
        #     print(f"Run {run + 1}: Inference Time = {inference_time_s:.6f} seconds = {inference_time_ms:.2f} milliseconds")
            
        #     inference_times.append(inference_time_s)

        # # 计算平均推理时间
        # average_inference_time_s = sum(inference_times) / num_runs  # 以秒为单位
        # average_inference_time_ms = average_inference_time_s * 1000  # 转换为毫秒

        # # 输出平均推理时间
        # print(f"\nAverage Inference Time: {average_inference_time_s:.6f} seconds")
        # print(f"Average Inference Time: {average_inference_time_ms:.2f} milliseconds")


    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
