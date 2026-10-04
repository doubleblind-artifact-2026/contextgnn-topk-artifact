from types import SimpleNamespace
import typing as t

from elliot.utils import logging
import logging as pylog

import math


class EarlyStopping:
    def __init__(self, early_stopping_ns: SimpleNamespace, validation_metric: str, validation_k: int, cutoffs: t.List,
                 simple_metrics: t.List, ):
        self.logger = logging.get_logger(self.__class__.__name__, pylog.DEBUG)
        self.validation_metric = validation_metric
        self.validation_k = validation_k
        self.cutoffs = cutoffs
        self.simple_metrics = simple_metrics
        self.monitor = getattr(early_stopping_ns, "monitor", self.validation_metric)

        if not len(early_stopping_ns.__dict__):
            self.active = False
        else:
            if not hasattr(early_stopping_ns, "patience"):
                self.patience = 0
            else:
                self.patience = early_stopping_ns.patience

            if self.monitor == "loss":
                if not hasattr(early_stopping_ns, "mode"):
                    self.mode = "min"
                elif early_stopping_ns.mode == "auto":
                    self.mode = "min"
                # observed_quantity = self._losses
                self.metric = False

            else:
                if not hasattr(early_stopping_ns, "mode"):
                    self.mode = "max"
                elif early_stopping_ns.mode == "auto":
                    self.mode = "max"

                metric = self.monitor.split("@")
                if metric[0].lower() not in [m.lower() for m in self.simple_metrics]:
                    raise Exception("Early stopping metric must be in the list of simple metrics")

                self.metric_k = int(metric[1]) if len(metric) > 1 else self.validation_k
                if self.metric_k not in self.cutoffs:
                    raise Exception("Validation cutoff must be in general cutoff values")
                self.metric = metric[0]
                # observed_quantity = [r[early_stopping_nsmetric_k]["val_results"][early_stopping_ns.metric] for r in self._results]

            if hasattr(early_stopping_ns, "min_delta"):
                self.min_delta = early_stopping_ns.min_delta

            if hasattr(early_stopping_ns, "rel_delta"):
                self.rel_delta = early_stopping_ns.rel_delta

            if hasattr(early_stopping_ns, "baseline"):
                self.baseline = early_stopping_ns.baseline

            self.verbose = getattr(early_stopping_ns, "verbose", False)
            self.active = True

    def stop(self, losses, results):
        if not self.active:
            return False
        
        if losses and (math.isnan(losses[-1]) or math.isinf(losses[-1])):
            return True
        
        if not self.metric:
            if not losses:
                return False
            
            current = losses[-1]
        else:
            if not results:
                return False
            
            last = results[-1]

            if (self.metric_k not in last or
                "val_results" not in last[self.metric_k] or
                self.metric not in last[self.metric_k]["val_results"]):

                return False
            
            current = last[self.metric_k]["val_results"][self.metric]

        if not hasattr(self, "_best"):
            self._best = current
            self._bad_epochs = 0

            return False

        min_delta = getattr(self, "min_delta", 0.0)
        rel_delta = getattr(self, "rel_delta", None)
        delta = abs(self._best) * rel_delta if rel_delta is not None else min_delta

        if self.mode == "max":
            improved = current > (self._best + delta)
        else:
            improved = current < (self._best - delta)

        if improved:
            self._best = current
            self._bad_epochs = 0

            return False
        else:
            self._bad_epochs += 1

            if self.verbose:
                self.logger.info(
                    f"EarlyStopping: current={current:.5f}, best={self._best:.5f}, "
                    f"bad_epochs={self._bad_epochs}/{self.patience}"
                )
                
            return self._bad_epochs >= self.patience

    def check_conditions(self, obs_0: float, obs_1: float):
        if hasattr(self, "min_delta") and hasattr(self, "rel_delta") and hasattr(self, "baseline"):
            return self.condition_base(obs_0, obs_1) \
                   or self.condition_min_delta(obs_0, obs_1) \
                   or self.condition_rel_delta(obs_0, obs_1) \
                   or self.condition_baseline(obs_0, obs_1)
        elif hasattr(self, "min_delta") and hasattr(self, "rel_delta"):
            return self.condition_base(obs_0, obs_1) \
                   or self.condition_min_delta(obs_0, obs_1) \
                   or self.condition_rel_delta(obs_0, obs_1)
        elif hasattr(self, "min_delta") and hasattr(self, "baseline"):
            return self.condition_base(obs_0, obs_1) \
                   or self.condition_min_delta(obs_0, obs_1) \
                   or self.condition_baseline(obs_0, obs_1)
        elif hasattr(self, "baseline") and hasattr(self, "rel_delta"):
            return self.condition_base(obs_0, obs_1) \
                   or self.condition_baseline(obs_0, obs_1) \
                   or self.condition_rel_delta(obs_0, obs_1)
        elif hasattr(self, "min_delta"):
            return self.condition_base(obs_0, obs_1) or self.condition_min_delta(obs_0, obs_1)
        elif hasattr(self, "rel_delta"):
            return self.condition_base(obs_0, obs_1) or self.condition_rel_delta(obs_0, obs_1)
        elif hasattr(self, "baseline"):
            return self.condition_base(obs_0, obs_1) or self.condition_baseline(obs_0, obs_1)
        else:
            return self.condition_base(obs_0, obs_1)

    def condition_base(self, obs_0: float, obs_1: float):
        if self.mode == 'min':
            return obs_0 < obs_1
        elif self.mode == 'max':
            return obs_0 > obs_1
        else:
            raise ValueError("mode option must be in the list [min, max, auto]")

    def condition_min_delta(self, obs_0: float, obs_1: float):
        return (obs_0 - obs_1) <= self.min_delta

    def condition_rel_delta(self, obs_0: float, obs_1: float):
        return (obs_0 - obs_1) <= obs_0 * self.rel_delta

    def condition_baseline(self, obs_0: float, obs_1: float):
        if self.mode == "min":
            return obs_0 >= self.baseline
        elif self.mode == "max":
            return obs_0 <= self.baseline
        else:
            raise ValueError("mode option must be in the list [min, max, auto]")

    def __str__(self):
        return ", ".join([f"{str(k)}: {str(v)}" for k, v in self.__dict__.items()])