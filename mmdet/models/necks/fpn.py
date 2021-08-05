import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        in_channels (List[int]): 输入的不同尺寸下的特征图的通道数量

        out_channels (int): Number of output channels (used at each scale)
        out_channels (int): 最终输出的特征图s的通道数，所有输出的特征图都是同一个输出通道数

        num_outs (int): Number of output scales.
        num_outs (int): 输出的特征图的数量

        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        start_level (int): 从backbone给出的inputs列表中的第几位开始计算。默认为0

        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        end_level (int): 从backbone给出的inputs列表中的第几位开始结束计算。默认为-1

        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # 对lateral层进行1*1的卷积，统一通道数。 是否使用norm可以设置
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1, # kernel_size
                conv_cfg=conv_cfg, #
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            # 对逐像素相加后的fpn特征图进行3*3的卷积，通道数不变
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3, # kernel_size
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        # 添加额外的卷积层
        extra_levels = num_outs - (self.backbone_end_level - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                # 1/2的下采样
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                # 加到了fpn的卷积层中
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()  # 自动精度切换
    def forward(self, inputs):
        """Forward function."""
        # 判定输入的特征图数量与限定的输入通道列表的元素数量一致
        assert len(inputs) == len(self.in_channels)

        # -------------------------------------- build laterals --------------------------------------
        # 创建横向连接， 每一个元素都是自底向上的横向连接的特征图。
        # 输入是原始C2,C3...的原始特征图，通道数也各异
        # 输出是L2，L3....，属于横向连接的特征图，通道数统一为d. 论文中为d=256
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # --------------------------------------  build top-down path --------------------------------------
        # 创建自顶向下的连接
        """
        F.interpolate的用法为：
            def interpolate(
                input: Tensor,
                size: Optional[int] = None,
                scale_factor: Optional[List[float]] = None,

                mode: str = 'nearest',
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None
            ) -> Tensor:
        其中size和scale_factor不共存，一个代表插值后的尺寸，一个代表缩放的比例，二者取一
        """
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            # 如果限定了缩放比例，就直接使用
            # 如果没有限定，就先取出低一层的特征图的尺寸

            # 直接在横向连接的特征图list中倒着向下扩大2倍，并逐像素相加
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(
                    laterals[i],
                    **self.upsample_cfg
                )
            else:
                prev_shape = laterals[i - 1].shape[2:]  # 取出低一层的特征图H和W
                laterals[i - 1] += F.interpolate(
                    laterals[i],
                    size=prev_shape,
                    **self.upsample_cfg
                )

        # -------------------------------------- build outputs --------------------------------------
        # part 1: from original levels
        # 使用3*3进行融合特征的平滑，通道数保持不变
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        # 如果要求的输出特征图数量超过了横向连接的数量，那么要在输出特征图list中继续补充
        """
        补充的方式有两种：
            1. 如果没有指定 "add_extra_convs"， 那就使用最大池化，卷积核尺寸为1，步长为2.  （为什么卷积核尺寸为1   ？？？？）
            2. 如果制定了 "add_extra_convs", 判断是哪种类型：
                （1）如果为"on_input",
                （2）如果为"on_lateral"
                （3）如果为"on_output"
        """
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            # 第一种： 使用max pool进行更高层级的信息提取  (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            # 第二种： 使用卷积层进行更高层级的信息提取
            else:
                # 先取出要使用的源特征图
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]  # 第一种：使用 backbone 给出的 C特征图
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]   # 第二种： 使用横向连接的最后一层特征图
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]  # 第三种： 使用输出特征图的最后一层特征图
                else:
                    raise NotImplementedError
                # 依次使用fpn convs进行进一步的特征图处理。 说明fpn convs的数量要等于最终输出的特征图数量
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # 还可以指定是否要使用relu
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
