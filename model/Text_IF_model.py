import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange
from torch.nn.init import trunc_normal_

class Text_IF(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                #  num_refinement_blocks=4,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Text_IF, self).__init__()

        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.encoder_B = Encoder_B(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)

        self.feature_fusion_4 = Fusion_Embed(embed_dim=dim * 2 ** 3)

        # self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 3)
        self.prompt_guidance_4 = ContextDecoder(visual_dim=dim * 2**3,transformer_width=512)
        
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.feature_fusion_3 = Fusion_Embed(embed_dim = dim * 2 ** 2)

        # self.prompt_guidance_3 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 2)
        self.prompt_guidance_3 = ContextDecoder(visual_dim=dim * 2 ** 2, transformer_width=512)


        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias) 
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.feature_fusion_2 = Fusion_Embed(embed_dim = dim * 2 ** 1)
        # self.prompt_guidance_2 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 1)
        self.prompt_guidance_2 = ContextDecoder(visual_dim=dim * 2 ** 1, transformer_width=512)



        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.feature_fusion_1 = Fusion_Embed(embed_dim = dim)
        # self.prompt_guidance_1 = FeatureWiseAffine(in_channels=512, out_channels=dim) # original
        self.prompt_guidance_1 = ContextDecoder(visual_dim=dim, transformer_width=512)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1
        # here no 1x1 conv to reduce channels
        self.decoder_level1 = nn.Sequential(*[ 
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.refinement = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refunet = RefUnet(in_ch=96, mid_ch=128) # PG-RRM
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias) # C=96 -> C=3
        
        ####################################################
        # self.text_chan_reduction1 = nn.Linear(in_features=512, out_features=dim) # denseclip in last layer
        # self.text_chan_reduction2 = nn.Linear(in_features=512, out_features=dim*2)
        # self.text_chan_reduction3 = nn.Linear(in_features=512, out_features=dim*2**2)
        # self.text_chan_reduction4 = nn.Linear(in_features=512, out_features=dim*2**3) # denseclip in first layer

        # self.gamma = nn.Parameter(torch.ones(1) * 1e-4)  # Initialize small like in DenseCLIP
        # self.score_projection = nn.Conv2d(1, 3, kernel_size=1)  # Project to RGB

        # self.attention_pool1 = AttentionPool2d( # for denseclip in last layer
        #     spacial_dim=96,  # Your spatial dimension (H=W=96)
        #     embed_dim=dim,  # Your channel dimension (dim*2)
        #     num_heads=4,  # small number bc of cuda memory
        #     output_dim=dim
        # )
        
        # self.attention_pool2 = AttentionPool2d(
        #     spacial_dim=48,
        #     embed_dim=dim*2,  # Change from dim to dim*2 to match checkpoint
        #     num_heads=4,
        #     output_dim=dim*2
        # )

        # self.attention_pool3 = AttentionPool2d(
        #     spacial_dim=24,
        #     embed_dim=dim*2**2,  # Change from dim to dim*2**2 to match checkpoint
        #     num_heads=4,
        #     output_dim=dim*2**2
        # )

        # self.attention_pool4 = AttentionPool2d( # for denseclip in first layer
        #     spacial_dim=12,  # Your spatial dimension (H=W=12)
        #     embed_dim=dim*2**3,  # Your channel dimension (dim*2**3)
        #     num_heads=4,  # small number bc of cuda memory
        #     output_dim=dim*2**3
        # )

        ####################################################


    def forward(self, inp_img_A, inp_img_B, text):
        b = inp_img_A.shape[0]
        text_features = self.get_text_feature(text.expand(b, -1)).to(inp_img_A.dtype)
        out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = self.encoder_A(inp_img_A) # original
        out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B = self.encoder_B(inp_img_B)

        out_enc_level4_A, out_enc_level4_B = self.cross_attention(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.feature_fusion_4(out_enc_level4_A, out_enc_level4_B)

        out_enc_level4 = self.attention_spatial(out_enc_level4)


        ##########################################################################
        # # For DenseCLIP loss
        # global_feat4, feature_map4 = self.attention_pool4(out_enc_level4) 
        # global_feat4 = global_feat4.unsqueeze(-1).unsqueeze(-1)
        # out_enc_level4, text_diff4 = self.prompt_guidance_4(out_enc_level4 + global_feat4, text_features) 
        # text4 = self.text_chan_reduction4(text_features) # [8, 512] -> [8, 384]
        # B, C, H, W = feature_map4.shape
        # text4 = text4.unsqueeze(1).expand(-1, 3, -1)  # B,C -> (B, 1, C) -> (B, 3, C) 
        # text_embeddings4 = text4 + self.gamma * text_diff4
        # B, K, C = text_embeddings4.shape
        # feature_map4 = F.normalize(feature_map4, dim=1, p=2)
        # text_normalized4 = F.normalize(text_embeddings4, dim=2, p=2)
        # score_map4 = torch.einsum('bchw,bkc->bkhw', feature_map4, text_normalized4)
        ## print(f"score_map4.shape: {score_map4.shape}") 
        # out_enc_level4 = torch.cat([out_enc_level4, score_map4], dim=1)
        ## print(f"out_enc_level4.shape after concat: {out_enc_level4.shape}")
        ###############################################################################

        out_enc_level4 = self.prompt_guidance_4(out_enc_level4, text_features) 
        inp_dec_level4 = out_enc_level4
        out_dec_level4 = self.decoder_level4(inp_dec_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4)

        inp_dec_level3 = self.prompt_guidance_3(inp_dec_level3, text_features)
        out_enc_level3 = self.feature_fusion_3(out_enc_level3_A, out_enc_level3_B)

        ##########################################################################
        # # For DenseCLIP loss for layer 3
        # global_feat3, feature_map3 = self.attention_pool3(out_enc_level3) # global_feat: [8, dim*2**2] = (b,c)
        # global_feat3 = global_feat3.unsqueeze(-1).unsqueeze(-1)
        # inp_dec_level3, text_diff3 = self.prompt_guidance_3(inp_dec_level3 + global_feat3, text_features)
        # text3 = self.text_chan_reduction3(text_features) # [8, 512] -> [8, dim*2**2]

        # B3, C3, H3, W3 = feature_map3.shape
        # text3 = text3.unsqueeze(1).expand(-1, 3, -1)  # B,C -> (B, 1, C) -> (B, 3, C)
        # text_embeddings3 = text3 + self.gamma * text_diff3
        # B3, K3, C3 = text_embeddings3.shape

        # feature_map3 = F.normalize(feature_map3, dim=1, p=2)
        # text_normalized3 = F.normalize(text_embeddings3, dim=2, p=2)
        # score_map3 = torch.einsum('bchw,bkc->bkhw', feature_map3, text_normalized3)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3, score_map3], dim=1)
        ##########################################################################

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        
        inp_dec_level2 = self.prompt_guidance_2(inp_dec_level2, text_features)
        out_enc_level2 = self.feature_fusion_2(out_enc_level2_A, out_enc_level2_B)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)

        ##########################################################################
        # # For DenseCLIP loss for layer 2
        # global_feat2, feature_map2 = self.attention_pool2(out_enc_level2) # global_feat: [8, dim*2] = (b,c)
        # global_feat2 = global_feat2.unsqueeze(-1).unsqueeze(-1)
        # inp_dec_level2, text_diff2 = self.prompt_guidance_2(inp_dec_level2 + global_feat2, text_features)
        # text2 = self.text_chan_reduction2(text_features) # [8, 512] -> [8, dim*2]

        # B2, C2, H2, W2 = feature_map2.shape
        # text2 = text2.unsqueeze(1).expand(-1, 3, -1)  # B,C -> (B, 1, C) -> (B, 3, C)
        # text_embeddings2 = text2 + self.gamma * text_diff2
        # B2, K2, C2 = text_embeddings2.shape

        # feature_map2 = F.normalize(feature_map2, dim=1, p=2)
        # text_normalized2 = F.normalize(text_embeddings2, dim=2, p=2)
        # score_map2 = torch.einsum('bchw,bkc->bkhw', feature_map2, text_normalized2)
        # inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2, score_map2], dim=1)
        ##########################################################################

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
         
        inp_dec_level1 = self.prompt_guidance_1(inp_dec_level1, text_features)
        out_enc_level1 = self.feature_fusion_1(out_enc_level1_A, out_enc_level1_B)
        ##########################################################################
        # # For DenseCLIP loss
        # global_feat1, feature_map1 = self.attention_pool1(out_enc_level1) # global_feat: [8, 48] = (b,c), feature_map: [8, 48, 96, 96] = (b,c,h,w)
        # global_feat1 = global_feat1.unsqueeze(-1).unsqueeze(-1)
        # inp_dec_level1, text_diff1 = self.prompt_guidance_1(inp_dec_level1 + global_feat1, text_features) # denseclip v3.1 out: [8, 48, 96, 96]
        # text1 = self.text_chan_reduction1(text_features) # [8, 512] -> [8, 48]

        # B1, C1, H1, W1 = feature_map1.shape
        # text1 = text1.unsqueeze(1).expand(-1, 3, -1)  # B,C -> (B, 1, C) -> (B, 3, C) # not sure if necessary
        # # also do it in the context decoder so can do it together
        # text_embeddings1 = text1 + self.gamma * text_diff1 # should i use this somewhere else too or just here?
        # B1, K1, C1 = text_embeddings1.shape

        # feature_map1 = F.normalize(feature_map1, dim=1, p=2)
        # text_normalized1 = F.normalize(text_embeddings1, dim=2, p=2)
        # score_map1 = torch.einsum('bchw,bkc->bkhw', feature_map1, text_normalized1)
        # inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1, score_map1], dim=1) # if comment this dont forget uncomment the cat later on
        # # sth here takes very long for val images
        ###########################################################
       
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1) 
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        # out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.refunet(out_dec_level1, text_features)
        out_dec = self.output(out_dec_level1)        
        return out_dec, text_features #, score_map1, score_map2, score_map3, score_map4
        

    @torch.no_grad()
    def  get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

## PG-RRM
class RefUnet(nn.Module):
    def __init__(self, in_ch=96, mid_ch=128, text_dim=512):
        super(RefUnet, self).__init__()

        # Initial projection from in_ch to mid_ch
        self.conv0 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)

        # Encoder path
        self.conv1 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(mid_ch, mid_ch*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_ch*2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(mid_ch*2, mid_ch*2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(mid_ch*2)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(mid_ch*2, mid_ch*4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(mid_ch*4)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Bottleneck
        self.conv5 = nn.Conv2d(mid_ch*4, mid_ch*4, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(mid_ch*4)
        self.relu5 = nn.ReLU(inplace=True)

        # Decoder path
        self.conv_d4 = nn.Conv2d(mid_ch*4 + mid_ch*4, mid_ch*2, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(mid_ch*2)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(mid_ch*2 + mid_ch*2, mid_ch*2, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(mid_ch*2)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(mid_ch*2 + mid_ch*2, mid_ch, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(mid_ch)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(mid_ch + mid_ch, mid_ch, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(mid_ch)
        self.relu_d1 = nn.ReLU(inplace=True)

        # Final output layer - back to input channel count for residual addition
        self.conv_d0 = nn.Conv2d(in_channels=mid_ch, out_channels=in_ch, kernel_size=3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # # Prompt guidance layers for encoder
        self.prompt_guidance_enc1 = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch)
        self.prompt_guidance_enc2 = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch*2)
        self.prompt_guidance_enc3 = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch*2)
        self.prompt_guidance_enc4 = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch*4)
        self.prompt_guidance_bottleneck = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch*4)
        
        # # Prompt guidance layers for decoder
        self.prompt_guidance_dec4 = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch*2)
        self.prompt_guidance_dec3 = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch*2)
        self.prompt_guidance_dec2 = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch)
        self.prompt_guidance_dec1 = FeatureWiseAffine(in_channels=text_dim, out_channels=mid_ch)
        self.prompt_guidance_out = FeatureWiseAffine(in_channels=text_dim, out_channels=in_ch)

    def forward(self, x, text_embed=None):
        # Input is 96 channels
        hx = x
        hx = self.conv0(hx)  # 96 -> mid_ch

        # Encoder path with text guidance
        hx1 = self.relu1(self.bn1(self.conv1(hx)))  # mid_ch -> mid_ch
        if text_embed is not None:
            hx1 = self.prompt_guidance_enc1(hx1, text_embed)
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))  # mid_ch -> mid_ch*2
        if text_embed is not None:
            hx2 = self.prompt_guidance_enc2(hx2, text_embed)
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))  # mid_ch*2 -> mid_ch*2
        if text_embed is not None:
            hx3 = self.prompt_guidance_enc3(hx3, text_embed)
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))  # mid_ch*2 -> mid_ch*4
        if text_embed is not None:
            hx4 = self.prompt_guidance_enc4(hx4, text_embed)
        hx = self.pool4(hx4)

        # Bottleneck with text guidance
        hx5 = self.relu5(self.bn5(self.conv5(hx)))  # mid_ch*4 -> mid_ch*4
        if text_embed is not None:
            hx5 = self.prompt_guidance_bottleneck(hx5, text_embed)

        # Decoder path with text guidance
        hx = self.upscore2(hx5)
        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))  # mid_ch*4 + mid_ch*4 -> mid_ch*2
        if text_embed is not None:
            d4 = self.prompt_guidance_dec4(d4, text_embed)
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))  # mid_ch*2 + mid_ch*2 -> mid_ch*2
        if text_embed is not None:
            d3 = self.prompt_guidance_dec3(d3, text_embed)
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))  # mid_ch*2 + mid_ch*2 -> mid_ch
        if text_embed is not None:
            d2 = self.prompt_guidance_dec2(d2, text_embed)
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))  # mid_ch + mid_ch -> mid_ch
        if text_embed is not None:
            d1 = self.prompt_guidance_dec1(d1, text_embed)

        residual = self.conv_d0(d1)  # mid_ch -> 96
        if text_embed is not None:
            residual = self.prompt_guidance_out(residual, text_embed)

        return x + residual

class ContextDecoder(nn.Module):
    def __init__(self,
            visual_dim,           # Number of channels in visual features (e.g., C = 384 for dim*2³)
            transformer_width=256, # Dimension of transformer's internal processing
            transformer_heads=4,   # Number of attention heads in transformer
            transformer_layers=6,  # Number of transformer decoder layers
            text_dim=512,         # Dimension of text features (e.g., 512 for CLIP)
            dropout=0.1,
            final_layer=False,
            **kwargs):
        

        super().__init__()

        self.final_layer = final_layer
        # Project visual features to transformer dimension
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        # Project text features to transformer dimension
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),  # Changed from visual_dim to text_dim
            nn.Linear(text_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads, dropout) 
            for _ in range(transformer_layers)
        ])
        
        if self.final_layer:
            # # Project back to visual dimension
            self.out_proj = nn.Sequential(
                nn.LayerNorm(transformer_width),
                nn.Linear(transformer_width, visual_dim)  # Output matches visual channels
            )


        self.MLP = nn.Sequential(
            nn.Linear(transformer_width, transformer_width * 2),
            nn.GELU(), # either this or in transformer was leaky relu
            nn.Linear(transformer_width * 2, visual_dim * 2)
        )


        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, visual, text):
        # text: [B, text_dim]
        # visual: [B, C, H, W]
        
        B, C, H, W = visual.shape
        
        # Reshape and project visual features
        visual_flat = visual.flatten(2).permute(0, 2, 1)  # [B, H*W, C]        
        visual_proj = self.memory_proj(visual_flat)  # [B, H*W, transformer_width]        

        # Project text
        if not self.final_layer:
            text = text.unsqueeze(1)  # [B, 1, text_dim]
        else:
            text = torch.cat([text.unsqueeze(1) for _ in range(3)], dim=1)  # [B, 3, text_dim]
    
        # Cross attention between text and visual features
        for layer in self.decoder:
            x = layer(text, visual_proj) # [B, K (=1 if final_layer), transformer_width]

        mlp_out = self.MLP(x) # [B, K, 2C]
        mlp_out = mlp_out.mean(dim=1, keepdim=True) # [B, 1, 2C]
        
        mlp_out = mlp_out.view(B, -1, 1, 1) # [B, 1, 2C] -> [B, 2C, 1, 1]
        gamma, beta = mlp_out.chunk(2, dim=1) # [B, 2C, 1, 1] -> [B, C, 1, 1], [B, C, 1, 1]
        out = (1+gamma)*visual + beta

        if self.final_layer:
            text_out = self.out_proj(x)
            return out, text_out
        
        return out # [B, C, H, W]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):  # x=text, mem=visual
   
        x = self.norm1(x)  # [B, N, C]
        mem = self.norm2(mem)  # [B, M, C]
        
        # Transpose for attention
        x = x.transpose(0, 1)  # [B, N, C] -> [N, B, C]
        mem = mem.transpose(0, 1)  # [B, M, C] -> [M, B, C]
        
        # Cross attention
        attn_output, _ = self.cross_attn(
            query=x,
            key=mem,
            value=mem
        )
        
        # Transpose back: [N, B, C] -> [B, N, C]
        x = (x + attn_output).transpose(0, 1)
        
        # MLP
        x = x + self.dropout(self.mlp(x)) 
        
        return x

class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

##########################################################################
## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1) # out: [B, 1, text_dim]
        batch = x.shape[0]
        if self.use_affine_level:
            mlp_out = self.MLP(text_embed)  # [B, 1, C*2]
            
            # Reshape to prepare for spatial broadcasting
            mlp_out = mlp_out.view(batch, -1, 1, 1)  # [B, C*2, 1, 1]
            
            gamma, beta = mlp_out.chunk(2, dim=1)  # Each: [B, C, 1, 1] 
            
            # Apply affine transformation:
            # - (1 + gamma) for scaling: initialized near 1 for stable training
            # - beta for shifting
            x = (1 + gamma) * x + beta  # [B, C, H, W] 
        return x


class Encoder_A(nn.Module):
    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_A, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_A, text_features=None):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)
        

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A)
        

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A)
        

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        out_enc_level4_A = self.encoder_level4(inp_enc_level4_A)
        

        return out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


class Encoder_B(nn.Module):
    def __init__(self, inp_channels=1, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_B, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_B, text_features=None):
        inp_enc_level1_B = self.patch_embed(inp_img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B)
        

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B)
        

        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B)
        

        inp_enc_level4_B = self.down3_4(out_enc_level3_B)
        out_enc_level4_B = self.encoder_level4(inp_enc_level4_B)
        

        return out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()
        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)
        
    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)

        return x

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, dense_clip_flag=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.dense_clip_flag = dense_clip_flag
       
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)