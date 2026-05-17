import contextlib
import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


def _autocast_ctx(amp_dtype, device):
    """Return an autocast context for the given dtype/device, or nullcontext.

    bfloat16 does NOT need a GradScaler (same exponent range as fp32), so we
    deliberately do not return one — callers can backprop directly. fp16 also
    skips GradScaler here for simplicity; if needed, callers should wrap their
    own training loop.
    """
    if amp_dtype is None:
        return contextlib.nullcontext()
    device_type = "cuda" if (isinstance(device, str) and "cuda" in device) or \
                            (hasattr(device, "type") and device.type == "cuda") else "cpu"
    return torch.amp.autocast(device_type=device_type, dtype=amp_dtype)


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True,
                 amp_dtype=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.amp_dtype = amp_dtype

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {
            metric.__name__: AverageValueMeter() for metric in self.metrics
        }

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                # Use __class__.__name__ to support both function and class-based losses
                loss_name = getattr(self.loss, '__name__', self.loss.__class__.__name__)
                loss_logs = {loss_name: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True,
                 amp_dtype=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
            amp_dtype=amp_dtype,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        with _autocast_ctx(self.amp_dtype, self.device):
            prediction = self.model.forward(x)
        # bfloat16 has the same exponent range as fp32, so no GradScaler is
        # needed; we just cast the prediction back to fp32 before the loss for
        # numerical stability (softmax / log_softmax / CE prefer fp32).
        prediction = prediction.float()
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True,
                 amp_dtype=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
            amp_dtype=amp_dtype,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            with _autocast_ctx(self.amp_dtype, self.device):
                prediction = self.model.forward(x)
            prediction = prediction.float()
            loss = self.loss(prediction, y)
        return loss, prediction
