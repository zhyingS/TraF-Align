from datasets.intermediate_fusion_dataset import getIntermediateFusionDataset
from datasets.no_fusion_dataset import getNoFusionDataset

from datasets.Basedataset.v2v4real_basedataset import V2V4RealBaseDataset
from datasets.Basedataset.v2xseq_basedataset import V2XSEQBaseDataset

__all__ = {
    "v2v4real": V2V4RealBaseDataset,
    "v2xseq": V2XSEQBaseDataset,
}


def build_dataset(dataset_cfg, set="train"):
    fusion_type = dataset_cfg["fusion"]["core_method"]
    dataset_name = dataset_cfg["fusion"]["dataset"]
    error_message = "This dataset is not supported."
    assert dataset_name in ["opv2v", "v2v4real", "v2xseq"], error_message

    if "Intermediate" in fusion_type:
        dataset = getIntermediateFusionDataset(__all__[dataset_name])(
            params=dataset_cfg, set=set
        )
    elif "NoFusion" in fusion_type:
        dataset = getNoFusionDataset(__all__[dataset_name])(params=dataset_cfg, set=set)
    else:
        raise (
            "The fusion type is not supported, should be IntermediateFusionDataset or NoFusionDataset."
        )

    return dataset
