"""Comfy memory-manager lease for an already resident text encoder.

Importing this module does not import torch or initialize CUDA. A closed lease is
never reloadable: the session creates a fresh PairedTextSession and lease.
"""
import weakref
import uuid


def _lease_class():
    import torch
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import CallbacksMP

    class TextPayload(torch.nn.Module):
        def __init__(self, device, dtype):
            super().__init__()
            self.device = device
            self.dtype = dtype

        def get_dtype(self):
            return self.dtype

    class ResidentTextLease(ModelPatcher):
        def __init__(self, session):
            paired = session.te
            encoder = paired.encoder
            device = torch.device(encoder.device)
            super().__init__(TextPayload(device, torch.float16), device, torch.device("cpu"))
            self._session = weakref.ref(session)
            self._paired = weakref.ref(paired)
            self._attached = []
            self.closed = False
            self._callback_key = "moss_text_lease:" + uuid.uuid4().hex

        def model_size(self):
            return self.loaded_size()

        def loaded_size(self):
            paired = self._paired()
            if self.closed or paired is None or paired.closed:
                return 0
            return int(paired.encoder.placement.resident_bytes())

        def model_patches_to(self, device):
            # The payload contains no weights or patches. TE owns its placement.
            return None

        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            if self.closed:
                return 0
            if device_to != self.load_device:
                raise RuntimeError("Resident TE lease cannot migrate devices")
            if force_patch_weights:
                raise RuntimeError("Resident TE lease does not accept weight patches")
            paired = self._paired()
            if paired is None or paired.closed:
                self._release()
                return 0
            self.model.device = self.load_device
            return 0  # Already resident; no upload occurred.

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            return self._release()

        def detach(self, unpatch_all=True):
            # MM re-registers an already-loaded model via detach(False), then
            # model_load. There are no patches to detach from this external owner.
            if unpatch_all:
                self._release()
            return self.model

        def clone(self, disable_dynamic=False, model_override=None, force_deepcopy=False):
            if model_override is not None or force_deepcopy:
                raise RuntimeError("Resident TE lease cannot duplicate its external weight owner")
            return self

        def _on_parent_clone(self, original, clone):
            if not self.closed:
                self.attach(clone)

        def attach(self, patcher):
            if self.closed:
                raise RuntimeError("Cannot attach a closed TE lease")
            patcher.set_additional_models("moss_text", [self])
            patcher.add_callback_with_key(CallbacksMP.ON_CLONE, self._callback_key, self._on_parent_clone)
            if not any(ref() is patcher for ref in self._attached):
                self._attached.append(weakref.ref(patcher))

        def _release(self):
            session = self._session()
            if session is None:
                return self._release_locked(None)
            with session.lock:
                return self._release_locked(session)

        def _release_locked(self, session):
            if self.closed:
                return 0
            freed = self.loaded_size()
            paired = self._paired()
            if paired is not None:
                paired.close()
            self.closed = True
            self.model.device = self.offload_device
            for ref in self._attached:
                patcher = ref()
                if patcher is not None:
                    patcher.remove_callbacks_with_key(CallbacksMP.ON_CLONE, self._callback_key)
                    remaining = [p for p in patcher.get_additional_models_with_key("moss_text") if p is not self]
                    if remaining:
                        patcher.set_additional_models("moss_text", remaining)
                    else:
                        patcher.remove_additional_models("moss_text")
            self._attached.clear()
            if session is not None:
                # MM may retain its LoadedModel after a successful partial unload.
                # Keep the tiny closed patcher alive until removal outside that callback.
                session.retired_te_leases.append(self)
                if session.te is paired:
                    session.te = None
                if session.te_lease is self:
                    session.te_lease = None
                session.counts["te_evictions"] = session.counts.get("te_evictions", 0) + 1
                session.last["TE_eviction"] = {"released_weight_bytes": freed, "reason": "Comfy ModelPatcher unload",
                                               "phase":getattr(session,"current_phase",None)}
            return freed

    return ResidentTextLease


def _drain_retired(session):
    from comfy import model_management as mm

    for retired in list(session.retired_te_leases):
        mm.unload_model_and_clones(retired, unload_additional_models=False)
        session.retired_te_leases.remove(retired)


def register_text_lease(session):
    """Register existing resident weights; call again after each TE recreation."""
    from comfy import model_management as mm

    with session.lock:
        if "retired_te_leases" not in vars(session):
            session.retired_te_leases = []
        _drain_retired(session)
        lease = session.te_lease
        if lease is not None and not lease.closed:
            if lease._paired() is not session.te:
                raise RuntimeError("Active TE lease belongs to another encoder")
            if session.patcher is not None:
                lease.attach(session.patcher)
            return lease
        if session.te is None or session.te.closed:
            raise RuntimeError("Load a resident text encoder before registering its lease")
        if session.te.encoder.plan.memory not in ("resident","partial"):
            raise ValueError("Text lease requires full or bounded partial residency")
        lease = _lease_class()(session)
        if lease.loaded_size() <= 0:
            raise ValueError("Resident text encoder has no device weights")
        session.te_lease = lease
        if session.patcher is not None:
            lease.attach(session.patcher)
        try:
            mm.load_models_gpu([lease])
        except Exception:
            lease._release()
            raise
        return lease


def close_text_lease(session):
    """Remove the standard MM entry before explicitly closing the session."""
    from comfy import model_management as mm

    with session.lock:
        if "retired_te_leases" not in vars(session):
            session.retired_te_leases = []
        _drain_retired(session)
        lease = session.te_lease
        if lease is not None:
            mm.unload_model_and_clones(lease, unload_additional_models=False)
            lease._release()
        _drain_retired(session)

