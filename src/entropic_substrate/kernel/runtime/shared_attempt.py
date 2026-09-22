"""One sequential provider/trainer lifecycle, specialized through domain adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn, cast


@dataclass(frozen=True, slots=True)
class AttemptFlow[T]:
    """Order irreversible transitions and dispatch every failure to its owner.

    Adapter callbacks hold their own typed state and preserve existing evidence
    contracts. Cleanup is required even when a finalization or billing step fails.
    """

    prepare: Callable[[], None]
    allocate: Callable[[], None]
    start: Callable[[], None]
    collect: Callable[[], None]
    terminate: Callable[[], None]
    bill: Callable[[], None]
    finalize: Callable[[], T]
    on_failure: Callable[[BaseException], NoReturn]

    def run(self) -> T:
        try:
            self.prepare()
            self.allocate()
            self.start()
            self.collect()
            self.terminate()
            self.bill()
            return self.finalize()
        except BaseException as error:
            # Untyped adapters must not turn a failed attempt into a normal return.
            cast(Callable[[BaseException], object], self.on_failure)(error)
            raise RuntimeError("attempt failure handler must not return") from error
