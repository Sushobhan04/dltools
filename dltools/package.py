import torch


def save_package(
    obj,
    filepath,
    library,
    package_name="model",
    resource_name="model.pkl",
    importer=None,
):
    if importer is None:
        importer = torch.package.sys_importer
    else:
        importer = (importer, torch.package.sys_importer)

    with torch.package.PackageExporter(filepath, importer=importer) as exp:
        for pkg in library["extern"]:
            exp.extern(pkg)

        for pkg in library["intern"]:
            exp.intern(pkg)

        for pkg in library["mock"]:
            exp.mock(pkg)

        exp.save_pickle(package_name, resource_name, obj)


def load_package(
    filepath,
    package_name="model",
    resource_name="model.pkl",
    map_location="cpu",
    return_importer=False,
):
    imp = torch.package.PackageImporter(filepath)
    obj = imp.load_pickle(package_name, resource_name, map_location=map_location)

    if return_importer:
        return obj, imp
    else:
        return obj


def load_model(filepath, with_optimizer=False, map_location="cpu"):
    data = torch.load(filepath, map_location=map_location)
    torch_model = load_package(data["torch_ckpt_path"])
    torch_model.load_state_dict(data["model_state_dict"])

    if with_optimizer:
        for k, v in torch_model.optimizer.items():
            v.load_state_dict(data[f"opt_{k}"])

    return torch_model
