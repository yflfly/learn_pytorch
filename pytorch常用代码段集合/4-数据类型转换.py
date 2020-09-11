# coding:utf-8
import torch

# 设置默认类型，pytorch中的FloatTensor远远快于DoubleTensor
torch.set_default_tensor_type(torch.FloatTensor)

# 类型转换
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()

# torch.Tensor与np.ndarray转换
# 除了CharTensor，其他所有CPU上的张量都支持转换为numpy格式然后再转换回来。

ndarray = tensor.cpu().numpy()
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float()  # If ndarray has negative stride.


