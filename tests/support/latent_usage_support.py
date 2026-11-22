"""The one registered latent-usage estimator for tests that drive the engine (FOR-1704, DH-52).

The runner, the engine and the worker hold no default instance; a test that runs them hands one
in, exactly as ``worker_command`` does at the composition seam.
"""

from entropic_substrate.core.sentence.scoring import LatentUsageOperand, latent_usage_estimator

POSTERIOR_MEAN = latent_usage_estimator(LatentUsageOperand.posterior_mean)
