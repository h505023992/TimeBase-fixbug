import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer

from thop import profile
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        # 创建一个输入张量（通常是模型的输入尺寸）
        input = torch.randn(1,args.seq_len ,args.enc_in ).to(self.device)
        # 使用 thop 计算 MACs 和参数数量
        macs, params = profile(self.model, inputs=(input,input[:,:,:6],input,input [:,:,:6]))
        print(f"MACs: {macs}")
        print(f"Params: {params}")
        if macs >= 1e9:
            print( f"{macs / 1e9:.2f}G MACs")
        elif macs >= 1e6:
            print( f"{macs / 1e6:.2f}M MACs")
        else:
            print( f"{macs} MACs")

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
