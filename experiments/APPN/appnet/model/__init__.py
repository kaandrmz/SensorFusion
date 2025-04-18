from .fpn import ResnetFPN
from .hrnet_ocr import HRNetW18_OCR, HRNetW48_OCR, HRNetW18, HRNetW48
from .pspnet import PSPNet
from .hrnet_ocr.hrnet import HRNetW48_OCR_1024
from .convnext_upernet import ConvNextUPNet_B,ConvNextUPNet_S,ConvNextUPNet_T,ConvNextUPNet_L,ConvNextUPNet_XL
from .condnet import CondNet_50,CondNet_101
from .vit_adapter import ViTAdapter_T,ViTAdapter_B,ViTAdapter_L,BeiTAdapter_L,MaskAdapter_L

NAME2MODEL = {"fpn": ResnetFPN, "psp": PSPNet, "hrnet18+ocr": HRNetW18_OCR, "hrnet48+ocr": HRNetW48_OCR,"hrnet18": HRNetW18, "hrnet48": HRNetW48,"hrnet48+ocr-1024":HRNetW48_OCR_1024,
              "convnextup_s":ConvNextUPNet_S,"convnextup_t":ConvNextUPNet_T,"convnextup_b":ConvNextUPNet_B,"convnextup_l":ConvNextUPNet_L,"convnextup_xl":ConvNextUPNet_XL,
              "condnet50":CondNet_50,"condnet101":CondNet_101,
              "vitadapter_t":ViTAdapter_T,"vitadapter_b":ViTAdapter_B,"vitadapter_l":ViTAdapter_L,"beitadapter_l":BeiTAdapter_L,"maskadapter_l":MaskAdapter_L}


def get_model_with_name(model_name):
    """Get model class with defined name

    Args:
        model_name (str): name of model

    Raises:
        ValueError: if not found the model

    Returns:
        torch.nn.Module class: model class
    """
    if model_name in NAME2MODEL:
        return NAME2MODEL[model_name]
    else:
        raise ValueError("Cannot found the implementation of model " + model_name)
