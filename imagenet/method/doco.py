from copy import deepcopy
from collections import deque
import math
import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import timm
from torchvision.datasets import ImageNet


from .vpt import PromptViT


class DOCO(nn.Module):
    def __init__(
        self,
        model: PromptViT,
        optimizer,
        ema_alpha=0.999,
        warmup_step=50,
        gmm_pool_size=512,
        reg_beta=0.5,
        lr=1e-1,
    ):
        super().__init__()

        self.ema_alpha = ema_alpha
        self.refine_step = 1
        self.warmup_step = warmup_step
        self.lr = lr
        self.reg_beta = reg_beta

        self.min_samples_for_adapt = 1

        assert hasattr(model, "vit"), "DOCO expects a PromptViT-wrapped model with .vit"
        assert hasattr(model.vit, "head"), "DOCO expects the wrapped ViT to have a classification head"
        
        self.model = model
        self.optimizer = optimizer

        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

        self.adapted_prompt_stats = None

        self.gmm_pool_size = gmm_pool_size
        self.ood_scores_pool = deque(maxlen=self.gmm_pool_size)

    def _has_adapted_prompt(self):
        return self.adapted_prompt_stats is not None

    def _get_saved_prompt_tensor(self):
        return self.adapted_prompt_stats[2].cuda()

    def _load_saved_prompt_into_model(self):
        self.model.prompts = torch.nn.Parameter(self._get_saved_prompt_tensor())

    def _build_prompt_optimizer(self):
        return torch.optim.AdamW([self.model.prompts], lr=self.lr)

    def _update_prompt_stats(self, batch_mean, batch_std):
        updated_prompt_tensor = self.model.prompts.detach().cpu()
        alpha = self.ema_alpha
        new_values = (batch_mean, batch_std, updated_prompt_tensor)
        for i, new_value in enumerate(new_values):
            self.adapted_prompt_stats[i] = (
                (1 - alpha) * self.adapted_prompt_stats[i]
                + alpha * new_value
            )
        
    def _compute_routing_features(self, x):
        raw_cls_features = self.model.forward_raw_features(x)[:, 0]
        prototypes = self.model.vit.head.weight.detach()

        if self._has_adapted_prompt():
            self._load_saved_prompt_into_model()
            features_prompted = self.model.forward_features(x)
            routing_cls_features = features_prompted[:, 0]
        else:
            routing_cls_features = raw_cls_features

        cos_sim = F.cosine_similarity(
            routing_cls_features.unsqueeze(1),
            prototypes,
            dim=2,
        )
        max_cos_sim, _ = cos_sim.max(1)
        return raw_cls_features, max_cos_sim

    def _predict_id_mask(self, ood_scores, device):
        current_ood_scores_np = ood_scores.cpu().numpy()
        self.ood_scores_pool.extend(current_ood_scores_np)

        pooled_scores = np.array(self.ood_scores_pool).reshape(-1, 1)
        km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(pooled_scores)

        filter_ids = km.predict(current_ood_scores_np.reshape(-1, 1))
        if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]:
            filter_ids = 1 - filter_ids

        id_mask = torch.from_numpy(filter_ids == 0).to(device)
        return id_mask

    def _split_batch(self, x):
        with torch.no_grad():
            raw_cls_features, max_cos_sim = self._compute_routing_features(x)

            ood_scores = (1 - max_cos_sim) * 100
            id_mask = self._predict_id_mask(ood_scores, x.device)

            x_id = x[id_mask]
            x_ood = x[~id_mask]
            raw_cls_features_id = raw_cls_features[id_mask]

        return {
            "id_mask": id_mask,
            "x_id": x_id,
            "x_ood": x_ood,
            "raw_cls_features_id": raw_cls_features_id,
        }
     
    def _compute_raw_id_alignment_loss(self, cls_features_id):
        
        src_std = self.train_info[0].cuda()
        src_mean = self.train_info[1].cuda()
        
        batch_mean = torch.mean(cls_features_id, dim=0)
        batch_std = torch.std(cls_features_id, dim=0)

        std_loss_raw = torch.norm(batch_std - src_std, p=2)
        mean_loss_raw = torch.norm(batch_mean - src_mean, p=2)
        loss_raw = std_loss_raw + mean_loss_raw
        return loss_raw, batch_mean, batch_std

    def _refine_existing_prompt(self, x_id, raw_cls_features_id):
        self.model.train()
        self.model.prompts = torch.nn.Parameter(self._get_saved_prompt_tensor())
        self.model.prompts.requires_grad_(True)
        self.optimizer = self._build_prompt_optimizer()

        outputs, loss = None, None
        batch_mean_adapted, batch_std_adapted = None, None

        raw_sim_matrix = None
        if x_id.shape[0] > 1 and self.reg_beta > 0:
            with torch.no_grad():
                raw_sim_matrix = pairwise_cosine_matrix(raw_cls_features_id)

        for _ in range(self.refine_step):
            outputs, loss, batch_mean_adapted, batch_std_adapted = forward_and_adapt(
                x_id,
                self.model,
                self.optimizer,
                self.train_info,
                raw_cls_features=raw_cls_features_id,
                reg_beta=self.reg_beta,
                raw_sim_matrix=raw_sim_matrix,
            )

        self._update_prompt_stats(
            batch_mean_adapted.detach().cpu(),
            batch_std_adapted.detach().cpu(),
        )
        return outputs, loss

    def _bootstrap_new_prompt(self, x_id, raw_cls_features_id, batch_mean, batch_std):
        self.model.train()
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        self.model.prompts.requires_grad_(True)
        self.optimizer = self._build_prompt_optimizer()

        outputs, loss = None, None
        raw_sim_matrix = None
        if x_id.shape[0] > 1 and self.reg_beta > 0:
            with torch.no_grad():
                raw_sim_matrix = pairwise_cosine_matrix(raw_cls_features_id)

        for _ in range(self.warmup_step):
            outputs, loss, _, _ = forward_and_adapt(
                x_id,
                self.model,
                self.optimizer,
                self.train_info,
                raw_cls_features=raw_cls_features_id,
                reg_beta=self.reg_beta,
                raw_sim_matrix=raw_sim_matrix,
            )

        self.adapted_prompt_stats = [
            batch_mean.clone().detach().cpu(),
            batch_std.clone().detach().cpu(),
            self.model.prompts.clone().detach().cpu(),
        ]
        return outputs, loss

    def _infer_id_without_adaptation(self, x_id):
        outputs_id = None
        self.model.eval()
        with torch.no_grad():
            if x_id.shape[0] > 0:
                outputs_id = self.model(x_id)
        return outputs_id

    def _infer_ood(self, x_ood):
        outputs_ood = None
        if x_ood.shape[0] > 0:
            self.model.eval()
            with torch.no_grad():
                outputs_ood = self.model(x_ood)
        return outputs_ood

    def _merge_outputs(self, outputs_id, outputs_ood, id_mask):
        if outputs_id is None and outputs_ood is None:
            return torch.tensor([]), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        if outputs_id is None:
            return outputs_ood
        if outputs_ood is None:
            return outputs_id

        out_dim = outputs_id.shape[1]
        final_outputs = torch.zeros(id_mask.shape[0], out_dim, device=id_mask.device)
        final_outputs[id_mask] = outputs_id
        final_outputs[~id_mask] = outputs_ood
        return final_outputs

    def forward(self, x):
        split_info = self._split_batch(x)

        x_id = split_info["x_id"]
        x_ood = split_info["x_ood"]
        raw_cls_features_id = split_info["raw_cls_features_id"]
        
        outputs_id, outputs_ood = None, None
        final_loss_raw = torch.tensor(0.0).cuda()
        final_loss_new = torch.tensor(0.0).cuda()
        final_loss_adapt = torch.tensor(0.0).cuda()

        if x_id.shape[0] > self.min_samples_for_adapt:
            
            loss_raw, batch_mean, batch_std = self._compute_raw_id_alignment_loss(raw_cls_features_id)
            loss_new = loss_raw

            has_adapted_prompt = self._has_adapted_prompt()

            final_loss_raw, final_loss_new = loss_raw, loss_new

            if has_adapted_prompt:
                outputs_id, final_loss_adapt = self._refine_existing_prompt(x_id, raw_cls_features_id)
            else:
                outputs_id, final_loss_adapt = self._bootstrap_new_prompt(x_id, raw_cls_features_id, batch_mean, batch_std)
        else:
            outputs_id = self._infer_id_without_adaptation(x_id)

        outputs_ood = self._infer_ood(x_ood)

        final_outputs = self._merge_outputs(
            outputs_id,
            outputs_ood,
            split_info["id_mask"],
        )

        if outputs_id is None and outputs_ood is None:
            return final_outputs

        return (
            final_outputs,
            final_loss_raw,
            final_loss_new,
            final_loss_adapt,
            split_info["id_mask"],
        )

    def obtain_src_stat(self, data_path, num_samples=300):
        num = 0
        features = []
        data_config = timm.data.resolve_model_data_config(self.model.vit)
        src_transforms = timm.data.create_transform(**data_config, is_training=False)

        src_dataset = ImageNet(
            root=f"{data_path}/ImageNet",
            split="train",
            transform=src_transforms,
        )
        src_loader = torch.utils.data.DataLoader(
            src_dataset,
            batch_size=64,
            shuffle=True,
        )

        with torch.no_grad():
            for img, _ in src_loader:
                images = img.cuda()
                raw_features = self.model.forward_raw_features(images)
                raw_logits = self.model.vit.forward_head(raw_features)

                ent = softmax_entropy(raw_logits)
                selected_indices = torch.where(ent < math.log(1000) / 2 - 1)[0]

                if selected_indices.numel() == 0:
                    continue

                selected_raw_cls = raw_features[selected_indices, 0]
                features.append(selected_raw_cls)

                num += selected_raw_cls.shape[0]
                if num >= num_samples:
                    break

            features = torch.cat(features, dim=0)
            features = features[:num_samples, :]
            self.train_info = torch.std_mean(features, dim=0)

        del features
    


    def reset(self):
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        self.adapted_prompt_stats = None
        self.ood_scores_pool.clear()



def pairwise_cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        x.unsqueeze(1),
        x.unsqueeze(0),
        dim=2,
    )

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def configure_model(model, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptViT(model, cfg.OPTIM.DOCO_PROMPT_NUM)
    for param in model.parameters():
        param.requires_grad_(False)
    model.prompts.requires_grad_(True)
    model.to(device)
    model.train()
    return model


def collect_params(model):
    return [model.prompts]


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


@torch.enable_grad()
def forward_and_adapt(x, model: PromptViT, optimizer, train_info, raw_cls_features, reg_beta, raw_sim_matrix=None):
    features_prompted = model.forward_features(x)
    cls_features_prompted = features_prompted[:, 0]

    batch_std, batch_mean = torch.std_mean(cls_features_prompted, dim=0)
    std_loss = torch.norm(batch_std - train_info[0].cuda(), p=2)
    mean_loss = torch.norm(batch_mean - train_info[1].cuda(), p=2)
    loss_stat = std_loss + mean_loss

    loss_reg = torch.tensor(0.0).cuda()
    if x.shape[0] > 1 and reg_beta > 0:
        sim_prompted = pairwise_cosine_matrix(cls_features_prompted)
        sim_raw = raw_sim_matrix if raw_sim_matrix is not None else pairwise_cosine_matrix(raw_cls_features)
        loss_reg = torch.norm(sim_prompted - sim_raw, p="fro")

    loss = loss_stat + reg_beta * loss_reg

    output = model.vit.forward_head(features_prompted)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output, loss, batch_mean, batch_std
