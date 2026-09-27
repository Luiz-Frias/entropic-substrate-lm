"""Reserve the same legal source-offset range across paired horizon arms."""

from dataclasses import replace

from entropic_substrate.core.corpus.batching import PredictiveBatch, SequenceSampler
from entropic_substrate.core.observation.contracts import ObservationAlignment


def paired_predictive_batch(
    sampler: SequenceSampler, horizon: int, *, reservation: int
) -> PredictiveBatch:
    """Sample the common range, then pass only the selected future to the model."""
    if type(horizon) is not int or type(reservation) is not int or not 1 <= horizon <= reservation:
        raise ValueError("horizon must be a positive integer within the shared reservation")
    batch = sampler.predictive_batch(reservation)
    if horizon == reservation:
        return batch
    time = batch.online_observations.shape[1]
    return replace(
        batch,
        target_encoder_context=batch.target_encoder_context[:, : time + horizon],
        alignments=tuple(
            ObservationAlignment.predictive(start=int(start), sequence_length=time, horizon=horizon)
            for start in batch.source_starts
        ),
        horizon=horizon,
    )
