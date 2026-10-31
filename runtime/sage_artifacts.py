"""Verified upstream Sage wheels selected by the host CUDA ABI family."""
BASE = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/"
ARTIFACTS = {
    12: {"filename": "sageattention-2.2.0+cu128torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl",
         "sha256": "103e06df49486daa87c3338bf2490ada2b074da1422f2b90566a8f2f81a6b76e", "bytes": 16871451},
    13: {"filename": "sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl",
         "sha256": "1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb", "bytes": 16656067},
}


def artifact(cuda):
    row = ARTIFACTS.get(int(str(cuda or "0").split(".")[0]))
    return dict(row, url=BASE + row["filename"].replace("+", "%2B")) if row else None
