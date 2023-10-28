import pathlib

import torch


class EasyDict(dict):
    @classmethod
    def from_dict(cls, d):
        out = cls()
        for k, v in d.items():
            if isinstance(v, dict) and not isinstance(v, EasyDict):
                out[k] = cls.from_dict(v)
            else:
                out[k] = v
        return out

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value) -> None:
        self[name] = value

    def __delattr__(self, name) -> None:
        del self[name]

    def to_dict(self):
        return to_dict(self)


def detach(data, cpu=False):
    if isinstance(data, torch.Tensor):
        _output = data.detach()
        if cpu:
            _output = _output.cpu()

    elif isinstance(data, EasyDict):
        _output = EasyDict()
        for k, v in data.items():
            _output[k] = detach(v)

    elif isinstance(data, dict):
        _output = {k: detach(v) for k, v in data.items()}

    elif isinstance(data, list):
        _output = [detach(item) for item in data]

    return _output


def to_device(data, device):
    _data = data
    if torch.is_tensor(data):
        _data = data.to(device)

    elif isinstance(data, list):
        _data = [to_device(item, device) for item in data]

    elif isinstance(data, EasyDict):
        _data = EasyDict()
        for k, v in data.items():
            _data[k] = to_device(v, device)

    elif isinstance(data, dict):
        _data = {k: to_device(v, device) for k, v in data.items()}

    return _data


def to_dict(data):
    if isinstance(data, (EasyDict, dict)):
        _data = {}
        for k, v in data.items():
            _data[k] = to_dict(v)
    elif isinstance(data, pathlib.Path):
        _data = data.as_posix()
    elif isinstance(data, tuple):
        _data = [to_dict(i) for i in data]
    else:
        _data = data

    return _data
