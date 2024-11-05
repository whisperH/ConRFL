from __future__ import absolute_import
from reid.utils.my_tools import *
import torch

from ..utils import to_torch

def extract_cnn_feature(model, inputs, args, middle_feat=False):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs).cuda()
        model_outputs = model(inputs)
        if not middle_feat:
            pass
            # if args.header == 'embedding':
            #     if args.use_TSBN:
            #         outputs = model_outputs['bn_feat']
            # elif args.header == 'sft':
            #     if args.use_TSBN:
            #         outputs = model_outputs['bn_feat']
            # elif args.header in part_header_list:
            #     bn_feat = model_outputs['bn_feat']
            #     bn_feat_part = model_outputs['bn_feat_part']
            #     outputs = torch.cat((bn_feat, bn_feat_part), 1)
            # outputs = outputs.data.cpu()
            # return outputs
        else:
            if args.header == 'embedding':
                outputs = model_outputs['bn_feat']
            elif args.header == 'partsft':
                outputs = model_outputs['bn_feat']
                channel_size = model_outputs['backbone_feat'].size(1)
                bn_backbone_feat = model_outputs['backbone_feat'].view(args.batch_size, channel_size, -1)
                model_outputs.update({
                    'bn_backbone_feat': model.module.bottleneck(bn_backbone_feat)
                })
            elif args.header in part_header_list:
                # channel_size = model_outputs['backbone_feat'].size(1)
                # bn_backbone_feat = model_outputs['backbone_feat'].view(args.batch_size, channel_size, -1)
                # model_outputs.update({
                #     'bn_backbone_feat': model.module.bottleneck(bn_backbone_feat)
                # })
                bn_feat = model_outputs['bn_feat']
                bn_feat_part = model_outputs['bn_feat_part']
                outputs = torch.cat((bn_feat, bn_feat_part), 1)

                # if args.header == 'midpart':
                if 'mid' in args.header:
                    recon_directions = model_outputs['recon_directions']
                    outputs = torch.cat((outputs, recon_directions), 1)
            outputs = outputs.data.cpu()

            model_outputs.update({
                'outputs': outputs
            })
            return model_outputs

def extract_pretain_feature(model, inputs):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs).cuda()
        model_outputs = model(inputs, use_pretrain_feat=True)
        model_outputs.update({
            'outputs': model_outputs['bn_feat']
        })
        return model_outputs