# utils/dino_similarity.py
import torch
import torch.nn.functional as F
import torchvision.transforms as T

class DinoSimilarity:
    def __init__(
        self,
        model_name="dinov2_vits14",
        device="cuda"
    ):
        self.device = device

        self.model = torch.hub.load(
            "dinov2/",
            model_name,
            source="local",
            pretrained=True
        ).to(self.device)

        self.model.eval()

        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

        self.patch_size = 14


    def extract_feats(self, image_bchw):
        """
        Args:
            image_bchw: [B, 3, H, W]

        Returns:
            features: [B, C, Hp, Wp]
        """

        h, w = image_bchw.shape[-2:]

        # DINO patch size의 배수가 되도록 조정
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size

        if new_h <= 0 or new_w <= 0:
            raise ValueError(
                f"Input image is too small: H={h}, W={w}"
            )

        img = F.interpolate(
            image_bchw,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        img = self.normalize(img)

        with torch.no_grad():
            patch_tokens = self.model.forward_features(
                img
            )["x_norm_patchtokens"]  # [B, N, C]

        hp = new_h // self.patch_size
        wp = new_w // self.patch_size

        features = self.patch_tokens_to_feature_map(
            patch_tokens,
            grid_height=hp,
            grid_width=wp
        )

        return features


    @staticmethod
    def patch_tokens_to_feature_map(
        patch_tokens,
        grid_height,
        grid_width
    ):
        """
        Args:
            patch_tokens: [B, N, C]

        Returns:
            feature_map: [B, C, Hp, Wp]
        """

        batch_size, num_tokens, feature_dim = (
            patch_tokens.shape
        )

        expected_tokens = grid_height * grid_width

        if num_tokens != expected_tokens:
            raise ValueError(
                f"Token count mismatch: "
                f"{num_tokens} != "
                f"{grid_height} x {grid_width}"
            )

        feature_map = (
            patch_tokens
            .transpose(1, 2)
            .reshape(
                batch_size,
                feature_dim,
                grid_height,
                grid_width
            )
            .contiguous()
        )

        return feature_map


    @staticmethod
    def upsample_features_then_cosine(
        render_features,
        gt_features,
        output_size,
        mode="bilinear",
        eps=1e-8
    ):
        """
        Patch-grid feature를 먼저 원본 해상도로 보간한 뒤
        각 픽셀 위치에서 cosine similarity를 계산한다.

        Args:
            render_features: [B, C, Hp, Wp]
            gt_features:     [B, C, Hp, Wp]
            output_size:     (H, W)

        Returns:
            similarity_map: [B, 1, H, W]
        """

        if render_features.shape != gt_features.shape:
            raise ValueError(
                f"Feature shape mismatch: "
                f"{render_features.shape} vs "
                f"{gt_features.shape}"
            )

        interpolate_kwargs = {
            "size": output_size,
            "mode": mode
        }

        # nearest 계열에서는 align_corners를 쓰면 안 됨
        if mode in ("linear", "bilinear", "bicubic", "trilinear"):
            interpolate_kwargs["align_corners"] = False

        render_dense = F.interpolate(
            render_features,
            **interpolate_kwargs
        )

        gt_dense = F.interpolate(
            gt_features,
            **interpolate_kwargs
        )

        # feature interpolation 이후 다시 정규화
        render_dense = F.normalize(
            render_dense,
            p=2,
            dim=1,
            eps=eps
        )

        gt_dense = F.normalize(
            gt_dense,
            p=2,
            dim=1,
            eps=eps
        )

        similarity_map = (
            render_dense * gt_dense
        ).sum(
            dim=1,
            keepdim=True
        )

        return similarity_map.clamp(-1.0, 1.0)


    def cosine_map(
        self,
        render_bchw,
        gt_bchw,
        dense_feature_upsampling=True,
        mode="bilinear"
    ):
        """
        Args:
            render_bchw: [B, 3, H, W]
            gt_bchw:     [B, 3, H, W]

        Returns:
            cosine map: [B, 1, H, W]
        """

        if render_bchw.shape[-2:] != gt_bchw.shape[-2:]:
            raise ValueError(
                f"Image size mismatch: "
                f"{render_bchw.shape[-2:]} vs "
                f"{gt_bchw.shape[-2:]}"
            )

        feat_r = self.extract_feats(render_bchw)
        feat_g = self.extract_feats(gt_bchw)

        output_size = gt_bchw.shape[-2:]

        if dense_feature_upsampling:
            # 추천 방식:
            # feature를 먼저 확대한 뒤 cosine 계산
            cosine = self.upsample_features_then_cosine(
                render_features=feat_r,
                gt_features=feat_g,
                output_size=output_size,
                mode=mode
            )
        else:
            # 기존 방식:
            # patch-grid cosine 계산 후 scalar map 확대
            cosine = F.cosine_similarity(
                feat_r,
                feat_g,
                dim=1
            ).unsqueeze(1)

            cosine = F.interpolate(
                cosine,
                size=output_size,
                mode=mode,
                align_corners=False
            )

        return cosine


    def norm_ratio_map(
        self,
        render_bchw,
        gt_bchw,
        eps=1e-8,
        dense_feature_upsampling=True,
        mode="bilinear"
    ):
        feat_r = self.extract_feats(render_bchw)
        feat_g = self.extract_feats(gt_bchw)

        output_size = gt_bchw.shape[-2:]

        if dense_feature_upsampling:
            render_dense = F.interpolate(
                feat_r,
                size=output_size,
                mode=mode,
                align_corners=False
            )

            gt_dense = F.interpolate(
                feat_g,
                size=output_size,
                mode=mode,
                align_corners=False
            )

            norm_r = torch.norm(
                render_dense,
                dim=1,
                keepdim=True
            )

            norm_g = torch.norm(
                gt_dense,
                dim=1,
                keepdim=True
            )

            ratio = norm_r / (norm_g + eps)

        else:
            norm_r = torch.norm(
                feat_r,
                dim=1,
                keepdim=True
            )

            norm_g = torch.norm(
                feat_g,
                dim=1,
                keepdim=True
            )

            ratio = norm_r / (norm_g + eps)

            ratio = F.interpolate(
                ratio,
                size=output_size,
                mode=mode,
                align_corners=False
            )

        return ratio
#########

# utils/dino_similarity.py
# import re
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T

# try:
#     from transformers import AutoImageProcessor, AutoModel
#     HF_AVAILABLE = True
# except Exception:
#     HF_AVAILABLE = False


# class DinoSimilarity:
#     def __init__(
#         self,
#         family="dinov2",                 # "dinov2" or "dinov3"
#         backend="local_hub",             # "local_hub" or "huggingface"
#         model_name=None,
#         repo_dir=None,                   # for local_hub
#         hf_model_id=None,                # for huggingface
#         weights_path=None,               # optional for local DINOv3 hub style
#         device="cuda",
#     ):
#         self.device = device
#         self.family = family.lower()
#         self.backend = backend.lower()

#         if self.family not in ["dinov2", "dinov3"]:
#             raise ValueError(f"Unsupported family: {family}")
#         if self.backend not in ["local_hub", "huggingface"]:
#             raise ValueError(f"Unsupported backend: {backend}")

#         if model_name is None:
#             model_name = "dinov2_vits14" if self.family == "dinov2" else "dinov3_vits16"
#         self.model_name = model_name

#         # HF 기본 모델명
#         if hf_model_id is None:
#             if self.family == "dinov2":
#                 hf_model_id = "facebook/dinov2-base"
#             else:
#                 hf_model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
#         self.hf_model_id = hf_model_id

#         # local hub 기본 repo
#         if repo_dir is None:
#             repo_dir = "dinov2/" if self.family == "dinov2" else "dinov3/"
#         self.repo_dir = repo_dir
#         self.weights_path = weights_path

#         self.patch_size = self._infer_patch_size()
#         self.processor = None
#         self.model = self._load_model().to(self.device)
#         self.model.eval()

#         # local_hub일 때만 torchvision normalize 사용
#         self.normalize = T.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225),
#         )

#     def _infer_patch_size(self):
#         # 예: dinov2_vits14, dinov3_vitb16-pretrain...
#         m = re.search(r'(\d+)', self.model_name)
#         if m is not None:
#             return int(m.group(1))
#         if self.hf_model_id is not None:
#             m2 = re.search(r'(\d+)', self.hf_model_id)
#             if m2 is not None:
#                 return int(m2.group(1))
#         return 14 if self.family == "dinov2" else 16

#     def _load_model(self):
#         if self.backend == "huggingface":
#             if not HF_AVAILABLE:
#                 raise ImportError("transformers가 설치되어 있어야 Hugging Face backend를 사용할 수 있습니다.")

#             self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
#             model = AutoModel.from_pretrained(self.hf_model_id)
#             return model

#         # local_hub backend
#         if self.family == "dinov2":
#             return torch.hub.load(
#                 self.repo_dir,
#                 self.model_name,
#                 source="local",
#                 pretrained=True
#             )

#         # local_hub + DINOv3
#         if self.weights_path is None:
#             raise ValueError("local_hub + dinov3 사용 시 weights_path가 필요합니다.")
#         return torch.hub.load(
#             self.repo_dir,
#             self.model_name,
#             source="local",
#             weights=self.weights_path
#         )

#     def _extract_feats_local_hub(self, image_bchw):
#         h, w = image_bchw.shape[-2:]
#         new_h = (h // self.patch_size) * self.patch_size
#         new_w = (w // self.patch_size) * self.patch_size

#         img = F.interpolate(image_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
#         img = self.normalize(img)

#         with torch.no_grad():
#             feats = self.model.forward_features(img)

#         if not isinstance(feats, dict) or "x_norm_patchtokens" not in feats:
#             raise KeyError("forward_features output에 'x_norm_patchtokens'가 없습니다.")

#         features = feats["x_norm_patchtokens"]  # [B, N, C]
#         hp, wp = new_h // self.patch_size, new_w // self.patch_size
#         c = features.shape[-1]
#         features = features.reshape(image_bchw.shape[0], hp, wp, c).permute(0, 3, 1, 2).contiguous()
#         return features

#     def _extract_feats_huggingface(self, image_bchw):
#         h, w = image_bchw.shape[-2:]
#         new_h = (h // self.patch_size) * self.patch_size
#         new_w = (w // self.patch_size) * self.patch_size

#         img = F.interpolate(image_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
#         img = torch.clamp(img, 0.0, 1.0)

#         # HF processor는 보통 CPU 입력을 기대하므로 한번 내립니다.
#         pixel_values = self.processor(
#             images=[img[i].detach().cpu() for i in range(img.shape[0])],
#             return_tensors="pt"
#         )["pixel_values"].to(self.device)

#         with torch.no_grad():
#             outputs = self.model(pixel_values=pixel_values)

#         # last_hidden_state: [B, 1+N, C] 또는 [B, N, C] 형태일 수 있음
#         tokens = outputs.last_hidden_state

#         # CLS 토큰이 있으면 제거
#         hp, wp = new_h // self.patch_size, new_w // self.patch_size
#         expected_n = hp * wp
#         if tokens.shape[1] == expected_n + 1:
#             tokens = tokens[:, 1:, :]
#         elif tokens.shape[1] != expected_n:
#             raise ValueError(
#                 f"Unexpected token count: got {tokens.shape[1]}, expected {expected_n} or {expected_n+1}"
#             )

#         c = tokens.shape[-1]
#         features = tokens.reshape(img.shape[0], hp, wp, c).permute(0, 3, 1, 2).contiguous()
#         return features

#     def extract_feats(self, image_bchw):
#         if self.backend == "huggingface":
#             return self._extract_feats_huggingface(image_bchw)
#         return self._extract_feats_local_hub(image_bchw)

#     def cosine_map(self, render_bchw, gt_bchw):
#         feat_r = self.extract_feats(render_bchw)
#         feat_g = self.extract_feats(gt_bchw)
#         cos = F.cosine_similarity(feat_r, feat_g, dim=1).unsqueeze(1)  # [B,1,Hp,Wp]
#         cos = F.interpolate(cos, size=gt_bchw.shape[-2:], mode="bilinear", align_corners=False)
#         return cos

#     def cosine_loss(self, render_bchw, gt_bchw):
#         cos = self.cosine_map(render_bchw, gt_bchw)
#         return (1.0 - cos).mean(), cos