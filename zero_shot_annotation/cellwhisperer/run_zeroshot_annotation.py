# start coding here
import anndata
import pandas as pd
import scipy
import torch
from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from cellwhisperer.config import get_path, model_path_from_name
from cellwhisperer.utils.inference import (
    score_transcriptomes_vs_texts,
    rank_terms_by_score,
    prepare_terms,
)

import cellwhisperer

from typing import Tuple
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from pathlib import Path
from cellwhisperer.utils.model_io import load_cellwhisperer_model


celltype_to_cell_ontology_name = {
    "Acinar": "pancreatic acinar cell",
    "Alpha": "pancreatic A cell",
    "Beta": "type B pancreatic cell",
    "Delta": "pancreatic D cell",
    "Epsilon": "pancreatic epsilon cell",
    "PP": "pancreatic PP cell",
    "pancreatic endocrine cell": "pancreatic endocrine cell",
    "Ductal": "pancreatic ductal cell",
    "PSC": "pancreatic stellate cell",

    "Endothelial": "endothelial cell",
    "endothelial cell": "endothelial cell",
    "lymphatic endothelial cell": "lymphatic endothelial cell",
    "endothelial cell of lymphatic vessel": "lymphatic endothelial cell",
    "endothelial tip cell": "endothelial tip cell",
    "endothelial stalk cell": "endothelial stalk cell",
    "endothelial cell of hepatic portal vein": "endothelial cell",
    "endothelial cell of periportal hepatic sinusoid": "endothelial cell of periportal hepatic sinusoid",
    "endothelial cell of pericentral hepatic sinusoid": "endothelial cell of pericentral hepatic sinusoid",

    "Mesenchymal": "mesenchymal cell",
    "mesenchymal cell": "mesenchymal cell",
    "stromal cell": "stromal cell",
    "stromal cell of lamina propria of colon": "stromal cell",
    "fibroblast": "fibroblast",
    "fibroblast cell": "fibroblast",
    "mesenchymal fibroblast": "fibroblast",
    "mesenchymal HAS1 high fibroblast": "fibroblast",
    "mesenchymal PLIN2+ fibroblast": "fibroblast",
    "myofibroblast": "myofibroblast",
    "mesenchymal myofibroblast": "myofibroblast",
    "kidney interstitial myofibroblast": "myofibroblast",
    "mesenchymal mesothelial cell": "mesothelial cell",
    "pericyte": "pericyte",
    "smooth muscle cell": "smooth muscle cell",
    "mesenchymal smooth muscle cell": "smooth muscle cell",
    "smooth muscle cell of large intestine": "smooth muscle cell",

    "hepatocyte": "hepatocyte",
    "hepatic stellate cell": "hepatic stellate cell",
    "Kupffer cell": "Kupffer cell",
    "cholangiocyte": "cholangiocyte",

    "B cell": "B cell",
    "mature B cell": "mature B cell",
    "T cell": "T cell",
    "T_cell": "T cell",
    "alpha-beta T cell": "alpha-beta T cell",
    "CD4-positive, alpha-beta T cell": "CD4-positive, alpha-beta T cell",
    "CD8-positive, alpha-beta T cell": "CD8-positive, alpha-beta T cell",
    "gamma-delta T cell": "gamma-delta T cell",
    "regulatory T cell": "regulatory T cell",
    "T follicular helper cell": "T follicular helper cell",
    "T-helper 17 cell": "T-helper 17 cell",
    "natural killer cell": "natural killer cell",
    "natural killer-like cell": "natural killer cell",
    "innate lymphoid cell": "innate lymphoid cell",
    "proliferating T cell": "cycling T cell",

    "myeloid cell": "myeloid cell",
    "monocyte": "monocyte",
    "Macrophage": "macrophage",
    "macrophage": "macrophage",
    "inflammatory macrophage": "inflammatory macrophage",
    "proliferating macrophage": "cycling macrophage",
    "conventional dendritic cell": "conventional dendritic cell",
    "plasmacytoid dendritic cell": "plasmacytoid dendritic cell",
    "neutrophil": "neutrophil",
    "Mast": "mast cell",
    "mast cell": "mast cell",

    "plasma cell": "plasma cell",
    "immune plasma cell": "plasma cell",
    "IgA plasma cell": "IgA plasma cell",

    "erythroid lineage cell": "erythroid lineage cell",
    "megakaryocyte": "megakaryocyte",
    "platelet": "platelet",

    "leukocyte cell": "kidney leukocyte cell",
    "epithelial cell": "epithelial cell",
    "MHC class II": "MHC class II-positive cell",
    "neuron": "neuron",
    "Schwann": "Schwann cell",

    "epithelial cell of large intestine": "epithelial cell of large intestine",
    "enterocyte of epithelium of large intestine": "colonocyte",
    "goblet cell": "goblet cell",
    "tuft cell of colon": "tuft cell",
    "glial cell": "enteroglial cell",
    "consensus molecular subtype 1 epithelial cell": "epithelial cell",
    "consensus molecular subtype 2 epithelial cell": "epithelial cell",
    "consensus molecular subtype 3 epithelial cell": "epithelial cell",

    "respiratory basal cell": "basal cell",
    "epithelial basal cell": "basal cell",
    "respiratory goblet cell": "goblet cell",
    "KRT8 expressing intermediate cell": "epithelial cell",
    "airway submucosal gland": "airway submucosal gland collecting duct epithelial cell",
    "lung ciliated cell": "ciliated cell",
    "epithelial ciliated cell": "ciliated cell",
    "epithelial differentiating ciliated cell": "ciliated cell",
    "hybrid secretory/ciliated cell": "epithelial cell",
    "epithelial SCGB3A2+ SCGB1A1+ cell": "club cell",
    "epithelial SCGB3A2+ cell": "secretory cell",
    "epithelial MUC5AC+ high": "goblet cell",
    "epithelial MUC5B+ cell": "mucus secreting cell",
    "lung neuroendocrine cell": "lung neuroendocrine cell",
    "pulmonary ionocyte": "pulmonary ionocyte",
    "epithelial alveolar type II cell": "pulmonary alveolar type 2 cell",
    "epithelial transitional alveolar type II cell": "pulmonary alveolar type 2 cell",
    "epithelial KRT5-/KRT17+ cell": "epithelial cell",
    "proliferating epithelial cell": "epithelial cell",
    "tuft-like cell": "tuft cell",

    "type a intercalated cell": "renal alpha-intercalated cell",
    "type b intercalated cell": "renal beta-intercalated cell",

    "nephron progenitor": "nephron progenitor cell",
    "metanephric cap": "nephron progenitor cell",
    "renal vesicle": "nephron tubule epithelial cell",
    "S shaped body": "nephron tubule epithelial cell",
    "ureteric bud": "ureteric bud cell",
    "epithelial cell of proximal tubule": "kidney proximal tubule epithelial cell",
    "proximal tubule cell": "kidney proximal tubule epithelial cell",
    "kidney loop of Henle epithelial cell": "kidney loop of Henle epithelial cell",
    "thick ascending limb of henle’s loop cell": "thick ascending limb of loop of Henle epithelial cell",
    "thick ascending limb of henle's loop cell": "thick ascending limb of loop of Henle epithelial cell",
    "distal convoluted tubule": "distal convoluted tubule epithelial cell",
    "connecting tubule and principal cell": "renal principal cell",
    "connecting nephron tubules/principal cell - proximal ureteric bud": "renal principal cell",
    "glomerular visceral epithelial cell": "podocyte",
    "renal filtration cell": "podocyte",
    "podocyte cell": "podocyte",
    "parietal epithelial cell": "parietal epithelial cell",
    "pelvic epithelium cell - distal ureteric bud": "urothelial cell",
    "uroepithelium cell": "urothelial cell",

    "Sertoli cell": "Sertoli cell",
    "spermatogonium": "spermatogonium",
    "spermatocyte": "spermatocyte",
    "spermatid": "spermatid",
    "Leydig cell and peritubular myoid cell": "Leydig cell",
    
    "B cell": "B cell",
  "Ciliated": "ciliated epithelial cell",
  "Endothelium": "endothelial cell",
  "Fibroblast": "fibroblast",
  "Lymphatic": "lymphatic endothelial cell",
  "Macrophages": "macrophage",
  "Mast cell": "mast cell",
  "NK cell": "natural killer cell",
  "Secretory": "secretory cell",
  "T cell": "T cell",
  "Transformed epithelium": "neoplastic epithelial cell",
  "Type 1": "pulmonary alveolar type 1 cell",
  "Type 2": "pulmonary alveolar type 2 cell",

  "B": "B cell",
  "Ery": "erythrocyte",
  "Mo": "monocyte",
  "Neu": "neutrophil",
  "NK": "natural killer cell",
  "T": "T cell"
}

def create_term_df(cell_types):
    """
    Create a DataFrame with a 'term' column containing the cell types. Mimicing the structure expected by the prepare_terms function. The table has 2 columns: library and term, where library is always "celltype" and term is the cell type name.
    """
    df = pd.DataFrame({"term": cell_types})
    df["library"] = "celltype"
    return df

# load model
modelpath = "/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/zero_shot_other_models/cellwhisperer/results/models/jointemb/cellwhisperer_clip_v1.ckpt"

# TODO replace model path

pl_model, tokenizer, transcriptome_processor = load_cellwhisperer_model(modelpath)

# dataset_names = ["E-CURD-114.project","E-GEOD-134144.project","E-HCAD-10.project","E-HCAD-15.project","E-HCAD-9.project","E-MTAB-5061","E-MTAB-8410.project","E-MTAB-9221.project","E-MTAB-9543.project","GSE185948","GSE81608","GSE84133","GSE85241","GSE86473", "GSE130148", "GSE149938"]

# dataset_names = ["GSE130148", "GSE149938"]
dataset_names = ['GSE111976']



   
for dts in dataset_names:
    print(f"Processing dataset {dts}...")
    adata = anndata.read_h5ad(f"/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/zero_shot_other_models/cellwhisperer/results/{dts}/cellwhisperer_clip_v1/leiden_umap_embeddings.h5ad")
    ref_adata = anndata.read_h5ad(f"/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/zero_shot_other_models/cellwhisperer/results/{dts}/cellwhisperer_clip_v1/cellxgene.h5ad", backed = 'r')

    ## makesure the cell id are consistent between the two adata
    assert all(adata.obs['orig_ids'] == ref_adata.obs_names), "Cell IDs do not match between the two adata files"

    adata.obs["index_int"] = list(range(len(adata.obs)))
    
    dfs = []

    cell_types = ref_adata.obs["cell_type"].unique()
    mapped_cell_types = [celltype_to_cell_ontology_name.get(ct, ct) for ct in cell_types]

    # candidate labels only need to be created once
    terms_df = create_term_df(mapped_cell_types)
    term_list = terms_df["term"].to_list()

    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()

    cell_embeds = torch.from_numpy(X).to(pl_model.device)

    scores, _ = score_transcriptomes_vs_texts(
        transcriptome_input=cell_embeds,
        text_list_or_text_embeds=term_list,
        logit_scale=pl_model.model.discriminator.temperature.exp(),
        model=pl_model.model,
        transcriptome_processor=transcriptome_processor,
        average_mode=None,
        score_norm_method="zscore",
    )

    pred_idx = scores.argmax(dim=0).cpu().numpy()
    pred_scores = scores.max(dim=0).values.cpu().numpy()

    similarity_scores_df = pd.DataFrame({
        "cell_index": ref_adata.obs_names,
        "leiden": adata.obs["leiden"].to_numpy() if "leiden" in adata.obs.columns else None,
        "true_label":ref_adata.obs["cell_type"].to_numpy(),
        "pred_label": terms_df["term"].iloc[pred_idx].to_numpy(),
        "pred_score": pred_scores,
    })

    ## save the similarity scores and predicted labels for each cell
    similarity_scores_df.to_csv(f"/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/zero_shot_other_models/cellwhisperer/results/{dts}/cellwhisperer_clip_v1/similarity_scores_per_cell.csv", index=False)